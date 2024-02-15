import torch
from torch import nn


class ActNorm(torch.nn.Module):
    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        super().__init__()

        self.register_buffer(
            "initialized", torch.tensor(False, dtype=torch.bool))
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    def reinitialize(self):
        self.initialized.data = torch.tensor(False, dtype=torch.bool)
        self.log_scale.data = torch.zeros_like(self.log_scale)
        self.shift.data = torch.zeros_like(self.shift)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def forward(self, inputs):
        if self.training and not self.initialized:
            self._initialize(inputs)

        scale, shift = self.scale.reshape(1, -1), self.shift.reshape(1, -1)
        outputs = scale * inputs + shift

        batch_size, _ = inputs.shape
        logabsdet = torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs):
        scale, shift = self.scale.reshape(1, -1), self.shift.reshape(1, -1)
        outputs = (inputs - shift) / scale

        batch_size, _ = inputs.shape
        logabsdet = -torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance."""
        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class TanhTransform(torch.nn.Module):
    def __init__(
        self,
        pre_act_scale: float,
        post_act_scale: float,
    ):
        super().__init__()
        self.pre_act_scale, self.post_act_scale = pre_act_scale, post_act_scale
        self.activation = torch.nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs * self.pre_act_scale
        outputs = self.activation(outputs)
        return outputs * self.post_act_scale

    def extra_repr(self) -> str:
        pre_act = "x"
        pre_act = (
            f"{pre_act} * {self.pre_act_scale}"
            if self.pre_act_scale != 1
            else f"{pre_act}"
        )
        act_name = "tanh"
        act = f"{act_name}({pre_act})"
        post_act = (
            f"{act} * {self.post_act_scale}" if self.post_act_scale != 1 else f"{act}"
        )
        return super().extra_repr() + f"forward: {post_act}"
