import torch
from .masked import MaskedAffineFlowTransform
from typing import Optional, List, Union, Tuple, Dict, Callable
from .postnonlinear import InPlaceTransform
import torch


class OSlow(torch.nn.ModuleList):
    def __init__(
        self,
        # architecture
        in_features: int,
        layers: List[int],
        dropout: Optional[float],
        residual: bool,
        activation: torch.nn.Module,
        # additional flow args
        additive: bool,
        num_transforms: int,
        normalization: Optional[Callable[[int], torch.nn.Module]],
        # base distribution
        base_distribution: torch.distributions.Distribution,
        # ordering
        ordering: Optional[torch.IntTensor],
        # post non linearity
        num_post_nonlinear_transforms: int = 0,
        **post_non_linear_transform_kwargs,
    ):
        """
        Args:
            in_features: number of input features
            layers: number of hidden units in the MLP of each layer
            residual: whether to use residual connections for the MLPs
            activation: activation function to use
            additive: whether to use additive coupling
            num_transforms: number of transformations to use
            normalization: normalization to use (e.g. ActNorm)
            base_distribution: base distribution to use (e.g. Normal)
            ordering: the autoregressive ordering of the model  (defaults to the identity ordering)
        """
        super().__init__()
        self.base_distribution = base_distribution
        self.in_features = in_features
        # initialize ordering
        self.register_buffer(
            "ordering",
            ordering
            if ordering is not None
            else torch.arange(
                self.in_features,
                dtype=torch.int,
            ),
        )
        if additive and num_transforms > 1:
            raise ValueError(
                "Cannot use additive coupling with more than one transform, this will turn the model affine!"
            )
        # instantiate flows
        for _ in range(num_transforms):
            self.append(
                MaskedAffineFlowTransform(
                    in_features=in_features,
                    layers=layers,
                    dropout=dropout,
                    residual=residual,
                    activation=activation,
                    additive=additive,
                    ordering=self.ordering,
                    normalization=normalization,
                )
            )

        for _ in range(num_post_nonlinear_transforms):
            self.append(
                InPlaceTransform(
                    shape=in_features,
                    normalization=normalization,
                    **post_non_linear_transform_kwargs,
                )
            )

    @property
    def device(self) -> torch.device:
        """
        Get the device of the model
        """
        return next(self.parameters()).device

    def forward(
        self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            inputs: samples from the data distribution (x's) of shape (batch_size, D)
            perm_mat: a batch of permutation matrices to use for the flow transforms (if None, then no the identity permutation is used) (batch_size, D, D)
        Returns: (z, log_dets)
            z: transformed samples from the base distribution (z's)
            log_dets: sum of the log determinants of the flow transforms
        """

        log_dets, z = 0, inputs
        for transform in self:
            if isinstance(transform, MaskedAffineFlowTransform):
                z, log_det = transform(inputs=z, perm_mat=perm_mat)
            elif isinstance(transform, InPlaceTransform):
                z, log_det = transform(inputs=z)
            else:
                raise ValueError("Unknown transform type")
            log_dets += log_det

        return z, log_dets

    def inverse(
        self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of the flow.

        Args:
            inputs: Batch of inputs to invert (results of a forward pass)
            perm_mat: The permutation for the autoregressive ordering

        Returns:
            Inverted inputs and log determinant of the inverse
        """
        z, log_dets = inputs, 0  # initialize z and log_dets
        # iterate over flows in reverse order and apply inverse

        for transform in reversed(self):
            if isinstance(transform, MaskedAffineFlowTransform):
                z, log_det = transform.inverse(inputs=z, perm_mat=perm_mat)
            elif isinstance(transform, InPlaceTransform):
                z, log_det = transform.inverse(inputs=z)
            else:
                raise ValueError("Unknown transform type")
            log_dets += log_det  # sign is handled in flow.inverse
        return z, log_det

    def reinitialize(self):
        for transform in self:
            if isinstance(transform, MaskedAffineFlowTransform):
                transform.reinitialize()
            elif isinstance(transform, InPlaceTransform):
                transform.reinitialize()
            else:
                raise ValueError("Unknown transform type")

    def sample(
        self, num_samples: int, perm_mat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample from the flow

        Args:
            num_samples: Number of samples to generate
        Returns:
            Samples
        """
        # Get the device of the current model
        device = self.device
        # Set the noises and set their device
        z = self.base_distribution.sample(
            (num_samples, self.in_features)).to(device)

        return self.inverse(z, perm_mat=perm_mat)[0]

    def log_prob(
        self, x: torch.Tensor, perm_mat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get log probability for batch

        $$\log p_x(x) = \log p_z(T(x)) + \log |det(J(T(x)))|$$

        Args:
            x: Batch of inputs of size (batch_size, D)
            perm_mat: The permutation for the autoregressive ordering
        Returns:
            log probabilities of size (batch_size,)
        """
        z, logabsdet = self.forward(x, perm_mat=perm_mat)
        log_base_prob = self.base_distribution.log_prob(z).sum(-1)
        return log_base_prob + logabsdet

    def intervene(
        self,
        num_samples: int,
        intervention: Dict[int, torch.Tensor],
        perm_mat: Optional[torch.Tensor] = None,
    ):
        """
        Intervene on the model by setting the value of some variables to a fixed value.

        Args:
            num_samples: The number of samples to generate from the interventional distribution
            intervention: A dictionary mapping node ids to their corresponding values
            perm_mat:
                The ordering that is being considered for the intervention.
                Defaults to None which is the identity ordering.
                **NOTE** The ordering must be the same for all samples in the batch.

        Returns:
            A batch of samples from the interventional distribution [batch_size, D]
        """
        # perm_mat is a tensor of size (num_samples, D, D)
        # raise exception if the permutation matrices of at least two samples are different
        if perm_mat is not None:
            perm_mat = perm_mat.reshape(-1,
                                        perm_mat.shape[-2], perm_mat.shape[-1])
            assert torch.all(
                perm_mat[0] == perm_mat
            ).item(), "Permutation matrices of at least two samples are different"
        else:
            perm_mat = (
                torch.eye(self.in_features)
                .unsqueeze(0)
                .repeat(num_samples, 1, 1)
                .to(self.device)
            )

        device = self.device
        # Set the noises and set their device
        base_z = self.base_distribution.sample((num_samples, self.in_features)).to(
            device
        )
        x = self.inverse(base_z, perm_mat=perm_mat)[0]

        entailed_ordering = torch.einsum(
            "bij, j -> bi", perm_mat, self.ordering).long()
        for i, idx in enumerate(entailed_ordering):
            if idx in intervention:
                x[:, idx] = intervention[idx]
                z = self(x, perm_mat=perm_mat)[0]
                z[:, entailed_ordering[i + 1:]] = base_z[:,
                                                         entailed_ordering[i + 1:]]
                x = self.inverse(z, perm_mat=perm_mat)[0]
        return x


# NOTE: this is a module for testing and we should remove it by the end
class OSlowTest(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        base_matrix: torch.Tensor,
    ):
        super().__init__()
        self.in_features = in_features
        self.dummy = torch.nn.Parameter(torch.randn(1))
        self.base_matrix = base_matrix

    @property
    def device(self) -> torch.device:
        """
        Get the device of the model
        """
        return next(self.parameters()).device

    def log_prob(
        self, x: torch.Tensor, perm_mat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if perm_mat.shape[0] != x.shape[0]:
            # repeat perm_mat to match the batch size
            perm_mat = perm_mat.unsqueeze(0).repeat(x.shape[0], 1, 1)

        dots = perm_mat * self.base_matrix[None, :, :].to(x.device)
        dots = dots.sum(-1).sum(-1) + self.dummy.detach() - self.dummy
        return dots
        # ret = []
        # for perm in perm_mat:
        #     key = '_'.join(perm.cpu().long().flatten().numpy().astype(str).tolist())
        #     ret.append(self.lookup_table[key])
        # ret = torch.tensor(ret).float().to(x.device) + self.dummy.detach() - self.dummy
        # return ret
