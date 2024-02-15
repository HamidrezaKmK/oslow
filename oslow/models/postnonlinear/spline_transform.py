from .splines import (
    unconstrained_rational_quadratic_spline,
    DEFAULT_MIN_BIN_HEIGHT,
    DEFAULT_MIN_BIN_WIDTH,
    DEFAULT_MIN_DERIVATIVE,
)
import warnings
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable


def _share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not (isinstance(x, int) and x >= 0):
        raise TypeError(
            "Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


class InPlaceTransform(nn.Module):
    def __init__(
        self,
        shape,
        num_bins=10,
        tail_bound=10.0,
        identity_init=False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        normalization: Optional[Callable[[int], torch.nn.Module]] = None,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        if normalization is not None:
            self.normalization = normalization(shape)
        else:
            self.normalization = None

        self.tail_bound = tail_bound

        if isinstance(shape, int):
            shape = (shape,)
        if identity_init:
            self.unnormalized_widths = nn.Parameter(
                torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(
                torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = num_bins - 1
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )
        else:
            self.unnormalized_widths = nn.Parameter(
                torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(
                torch.rand(*shape, num_bins))

            num_derivatives = num_bins - 1
            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    def reinitialize(self):
        if self.normalization:
            self.normalization.reinitialize()
        if isinstance(self.unnormalized_widths, nn.Parameter):
            nn.init.xavier_normal_(self.unnormalized_widths)
        if isinstance(self.unnormalized_heights, nn.Parameter):
            nn.init.xavier_normal_(self.unnormalized_heights)
        if isinstance(self.unnormalized_derivatives, nn.Parameter):
            nn.init.xavier_normal_(self.unnormalized_derivatives)

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = _share_across_batch(
            self.unnormalized_widths, batch_size)
        unnormalized_heights = _share_across_batch(
            self.unnormalized_heights, batch_size
        )
        unnormalized_derivatives = _share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        outputs, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            tail_bound=self.tail_bound,
        )

        return outputs, torch.sum(logabsdet, dim=1)

    def forward(self, inputs):
        inputs, logdets = self._spline(inputs, inverse=False)
        if self.normalization:
            inputs, logdets_ = self.normalization(inputs)
            logdets = logdets + logdets_
        return inputs, logdets

    def inverse(self, inputs):
        logdets = 0
        if self.normalization:
            inputs, logdets = self.normalization.inverse(inputs)
        inputs, logdets_ = self._spline(inputs, inverse=True)
        logdets = logdets + logdets_
        return inputs, logdets
