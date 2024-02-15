import torch
from typing import List, Optional, Union
from .linear import MaskedLinear


class MaskedBlock(MaskedLinear):
    def __init__(
        self,
        in_features: List[int],
        out_features: List[int],
        residual: bool,
        activation: Optional[torch.nn.Module],
        dropout: Optional[float],
        auto_connection: bool,
        ordering: Optional[torch.IntTensor],
    ):
        # init linear
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            auto_connection=auto_connection,
            ordering=ordering,
        )

        if residual and not auto_connection:
            raise ValueError(
                "residual connections are only supported for auto_connection=True"
            )

        self.residual = residual
        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None
        self.activation = activation

    def reinitialize(self):
        super().reinitialize()
        if self.dropout:
            self.dropout.reset_parameters()

    def forward(self, inputs: torch.Tensor, perm_mat: torch.Tensor) -> torch.Tensor:
        outputs = super().forward(inputs, perm_mat=perm_mat)
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.dropout(outputs) if self.dropout else outputs
        if self.residual:
            outputs = outputs + inputs
        return outputs


class MaskedMLP(torch.nn.ModuleList):
    def __init__(
        self,
        in_features: int,
        layers: List[int],
        dropout: Optional[float],
        residual: bool,
        activation: torch.nn.Module,
        ordering: torch.IntTensor,
    ):
        """
        Initialize a Masked (autoregressive) Multi-Layer Perceptron.
        """
        super().__init__()
        # process arguments (put them in the right format)
        layers += [1]
        in_features = [1] * in_features
        layers = [[layer] * len(in_features) for layer in layers]
        # set architectural hyperparameters
        self.in_features = in_features
        # initialize layers
        for i in range(len(layers)):
            self.append(
                MaskedBlock(
                    in_features=in_features if i == 0 else layers[i - 1],
                    out_features=layers[i],
                    dropout=dropout,
                    activation=activation if i < len(layers) - 1 else None,
                    auto_connection=True if i < len(layers) - 1 else False,
                    residual=residual if i < len(layers) - 1 else False,
                    ordering=ordering,
                )
            )

    def reinitialize(self):
        for layer in self:
            layer.reinitialize()

    def forward(
        self,
        inputs,
        perm_mat: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model. This function is called automatically when the model is called.

        Args:
            inputs: The inputs to the model. The inputs should be a tensor of shape (batch_size, in_features) or
                a tensor of shape (batch_size, *) and force_reshape_inputs=True.
            perm_mat: The permutation matrix to use for the inputs. If None, the inputs are not permuted.
                One can use this argument to learn the permutation matrix for instance using the sinkhorn operator.
            mask_index: The index of the mask to use. (see `reorder` function for more details)
                by default the current ordering of the model is used. (reorder should be called manually to move
                between orderings)

        Returns:
            The output of the model.
        """
        for layer in self:
            inputs = layer(inputs, perm_mat=perm_mat)
        return inputs
