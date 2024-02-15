import torch
from typing import List, Optional, Union
import functools
import math


class MaskedLinear(torch.nn.Linear):
    """
    Linear layer with ordered outputs to maintain an autoregressive data flow.

    the output neurons are ordered in blocks. Each block can
    have a different number of neurons. The number of neurons in each block
    is determined how `out_features` is specified.

    For example, if `in_features` is [3, 4, 2, 3] and `out_features` is
    [10, 5, 7, 3], then the output neurons are ordered in blocks of size
    10, 5, 7, and 3. If a natural ordering is used, then the first 10 neurons
    will be labeled 0, and the next 5 will be labeled 1, and so on.
    """

    def __init__(
        self,
        in_features: List[int],
        out_features: List[int],
        auto_connection: bool,
        ordering: Optional[torch.IntTensor],
    ) -> None:
        """
        A linear layer with support for handling dependencies between input and output.
        Each input or output feature has a label and two features are connected if the
        output feature has a label greater than the input feature w.r.t. the ordering.

        Args:
            in_features:
                list of integers where the i'th integer represents the number of input features that have label i.
            out_features:
                list of integers where the i'th integer represents the number of output features that have label i.
            auto_connection: whether to allow equal label connections.
            ordering: the custom ordering to use for the labels. If None, then the identity ordering is used.
        """
        self.in_features = in_features
        self.out_features = out_features

        if len(in_features) != len(out_features):
            raise ValueError(
                "in_features and out_features must have the same length")

        if ordering is None:
            ordering = torch.arange(
                len(in_features),
                dtype=torch.int,
            )

        # setup underlying linear layer
        total_in_features = sum(in_features)
        total_out_features = sum(out_features)
        super().__init__(
            in_features=total_in_features,
            out_features=total_out_features,
        )

        d = len(ordering)
        # change the permutation such that ordering_i is the location where i appears in ordering
        ordering = torch.argsort(ordering)
        # Expand ordering to a 2D tensor by repeating it along rows and columns
        ordering_row = ordering.unsqueeze(1).expand(d, d)
        ordering_col = ordering.unsqueeze(0).expand(d, d)

        diagonal = (
            (ordering_row == ordering_col).int()
            if auto_connection
            else torch.zeros(d, d, dtype=torch.int)
        )
        self.register_buffer(
            "mask",
            (ordering_row > ordering_col).int() + diagonal,
        )

    def reinitialize(self):
        """
        Reinitialize the weights and biases of the linear layer.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        inputs: torch.Tensor,
        perm_mat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes masked linear operation.

        Inputs are expected to be batched in the first dimension (2D tensor), or
        a result of a previous masked linear operation (3D tensor, for perm_mat with
        multiple permutations). If the input is a 3D tensor, the dimensions are
        expected to be (num_perms, batch, in_features).

        Args:
            inputs: Input tensor (batched in the first dimensions.) (b, D)
            perm_mat: Permutation matrix for the output neurons. ([num_perms], D x D)

        Returns:
            A `torch.Tensor` which equals to masked linear operation on inputs, or:
                `inputs @ (self.mask * self.weights).T + self.bias`
            output shape is (batch, out_features) if perm_mat is not None,
        """
        d = self.mask.shape[0]
        perm_mat_ = torch.eye(d) if perm_mat is None else perm_mat.clone()
        if perm_mat_.dim() == 2:
            perm_mat_ = perm_mat_.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        perm_mat_ = perm_mat_.to(self.weight.device)

        # compute the mask and mask the weights

        mask = torch.einsum(
            "bij,jk,bkl->bil", perm_mat_, self.mask.float(), perm_mat_.transpose(-2, -1)
        )

        # we have to populate the mask with the correct number of features per dimension
        mask = torch.repeat_interleave(
            mask, torch.tensor(self.in_features // d).to(mask.device), dim=-1
        )
        mask = torch.repeat_interleave(
            mask, torch.tensor(self.out_features // d).to(mask.device), dim=-2
        )

        ret = torch.einsum("bij,bj->bi", mask *
                           self.weight, inputs) + self.bias

        return ret
