import typing as th
import torch


class InterventionChainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n: int,
        num_samples: int = 1000,
        base_distribution: str = "torch.distributions.Normal",
        base_distribution_args: dict = dict(loc=0.0, scale=1.0),
        dislocate: bool = False,
        transient: bool = True,
        seed: int = 0,
        weight_s: th.Union[th.Tuple[float, float], float] = 0.1,
        weight_t: th.Union[th.Tuple[float, float], float] = 0.1,
    ):
        super().__init__()
        self.base_distribution = dy.get_value(base_distribution)(
            **base_distribution_args
        )
        self.num_samples = num_samples
        self.n = n
        self.seed = seed
        self.dislocate = dislocate
        self.transient = transient
        torch.manual_seed(seed)
        if isinstance(weight_s, (list, tuple)):
            self.s_weight = torch.rand(n) * (weight_s[1] - weight_s[0]) + weight_s[0]
        else:
            self.s_weight = torch.ones(n) * weight_s
        if isinstance(weight_t, (list, tuple)):
            self.t_weight = torch.rand(n) * (weight_t[1] - weight_t[0]) + weight_t[0]
        else:
            self.t_weight = torch.ones(n) * weight_t

        self.data, self.means, self.stds = self._generate_data()

    def s_func(self, x):
        return torch.nn.functional.softplus(x)

    def t_func(self, x):
        return x + torch.sin(x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def _generate_data(self):
        base_noise = self.base_distribution.sample((self.num_samples, self.n))
        means, stds = torch.empty(self.n), torch.empty(self.n)
        for i in range(self.n):
            noise = base_noise[:, i]
            if self.transient:
                scale = (
                    torch.ones(self.num_samples)
                    if i == 0
                    else (results * self.s_weight[:i]).sum(dim=1)
                )
                transform = (
                    torch.zeros(self.num_samples)
                    if i == 0
                    else (results * self.t_weight[:i]).sum(dim=1)
                )
            else:
                scale = (
                    torch.ones(self.num_samples)
                    if i == 0
                    else (results[:, i - 1] * self.s_weight[i - 1])
                )
                transform = (
                    torch.zeros(self.num_samples)
                    if i == 0
                    else (results[:, i - 1] * self.t_weight[i - 1])
                )

            x_i = noise * self.s_func(scale) + self.t_func(transform)
            means[i] = x_i.mean()
            stds[i] = x_i.std()
            if self.dislocate:
                x_i = (x_i - means[i]) / stds[i]
            results = (
                torch.cat([results, x_i.reshape(-1, 1)], dim=1)
                if i > 0
                else x_i.reshape(-1, 1)
            )
        return results, means, stds

    def intervene(self, idx, value, num_samples=1000):
        base_noise = self.base_distribution.sample((num_samples, self.n))
        for i in range(self.n):
            if i == idx:
                x_i = (torch.ones(num_samples) * value).reshape(-1, 1)
            else:
                noise = base_noise[:, i]
                if self.transient:
                    scale = (
                        torch.ones(self.num_samples)
                        if i == 0
                        else (results * self.s_weight[:i]).sum(dim=1)
                    )
                    transform = (
                        torch.zeros(self.num_samples)
                        if i == 0
                        else (results * self.t_weight[:i]).sum(dim=1)
                    )
                else:
                    scale = (
                        torch.ones(self.num_samples)
                        if i == 0
                        else (results[:, i - 1] * self.s_weight[i - 1])
                    )
                    transform = (
                        torch.zeros(self.num_samples)
                        if i == 0
                        else (results[:, i - 1] * self.t_weight[i - 1])
                    )
                x_i = noise * self.s_func(scale) + self.t_func(transform)
                if self.dislocate:
                    x_i = (x_i - self.means[i]) / self.stds[i]
            results = (
                torch.cat([results, x_i.reshape(-1, 1)], dim=1)
                if i > 0
                else x_i.reshape(-1, 1)
            )
        return results

    def do(
        self,
        idx,
        values: th.Union[torch.Tensor, list],
        target: th.Optional[int] = None,
        num_samples=50,
    ):
        values = (
            values.reshape(-1).tolist() if isinstance(values, torch.Tensor) else values
        )
        results = torch.stack(
            [
                self.intervene(idx=idx, value=value, num_samples=num_samples)
                for value in values
            ],
            dim=0,
        )
        return results[:, :, target] if target is not None else results
