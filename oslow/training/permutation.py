import torch
import abc

from oslow.models import OSlow
from oslow.training.utils import sample_gumbel_noise, hungarian, turn_into_matrix
from typing import Optional, Literal, List


def _sinkhorn(log_x: torch.Tensor, iters: int, temp: float):
    """
    Perform a sequence of normalizations to make the matrix doubly stochastic
    This is known as the Sinkhorn operator.
    When the number of iterations are large, this value converges to the Hungarian algorithm's result.
    """
    n = log_x.size()[1]
    log_x = log_x.reshape(-1, n, n) / temp
    for _ in range(iters):
        log_x = log_x - (torch.logsumexp(log_x, dim=2,
                         keepdim=True)).reshape(-1, n, 1)
        log_x = log_x - (torch.logsumexp(log_x, dim=1,
                         keepdim=True)).reshape(-1, 1, n)
    results = torch.exp(log_x)
    return results


class PermutationLearningModule(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        in_features: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.fixed = False

    def fix_permutation(
        self,
        perm: List[int],
    ):
        self.fixed = True
        self.fixed_perm = torch.IntTensor(perm)

    def release_fix(
        self,
    ):
        self.fixed = False

    @property
    def gamma(self):
        return self._gamma

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        raise NotImplementedError

    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0) -> torch.Tensor:
        raise NotImplementedError

    def get_best(self, temperature: float = 1.0):
        permutation_samples = self.sample_hard_permutations(
            100, unique_and_resample=False, gumbel_std=temperature).detach()
        # find the majority of permutations being sampled
        permutation, counts = torch.unique(
            permutation_samples, dim=0, return_counts=True
        )
        # find the permutation with the highest count
        return permutation[counts.argmax()]


class SoftSort(PermutationLearningModule):
    def __init__(
            self,
            in_features: int,
            set_gamma_uniform: bool = False,
            set_gamma_custom: Optional[List[List[int]]] = None,
            *args,
            temp: float = 0.1,
            **kwargs
    ):
        super().__init__(in_features=in_features, *args, **kwargs)
        if set_gamma_uniform:
            self.register_parameter(
                "_gamma", torch.nn.Parameter(torch.ones(in_features)))
        elif set_gamma_custom is not None:
            all_perms = []
            for perm in set_gamma_custom:
                all_perms.append(torch.tensor(perm).float())
            all_perms = torch.stack(all_perms)
            self.register_parameter(
                "_gamma", torch.nn.Parameter(torch.mean(all_perms, dim=0))
            )
        else:
            self.register_parameter(
                "_gamma", torch.nn.Parameter(torch.randn(in_features)))
        self.temp = temp

    def sample_hard_permutations(
        self, num_samples: int, return_noises: bool = False, unique_and_resample: bool = False, gumbel_std: float = 1.0
    ):
        if self.fixed:
            return self.fixed_perm.unsqueeze(0).repeat(num_samples, 1)
        else:
            gumbel_noise = sample_gumbel_noise(
                (num_samples, *self.gamma.shape),
                device=self.gamma.device,
                std=gumbel_std,
            )
            scores = self.gamma + gumbel_noise
            # perform argsort on every line of scores
            permutations = torch.argsort(scores, dim=-1)
            if unique_and_resample:
                permutations = torch.unique(permutations, dim=0)
                permutations = permutations[
                    torch.randint(
                        0,
                        permutations.shape[0],
                        (num_samples,),
                        device=permutations.device,
                    )
                ]

            # turn permutations into permutation matrices
            ret = torch.stack([turn_into_matrix(perm)
                              for perm in permutations])

            # add some random noise to the permutation matrices
            if return_noises:
                return ret, gumbel_noise
            return ret

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_hard_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        scores = self.gamma + gumbel_noise
        all_ones = torch.ones_like(scores)
        scores_sorted = torch.sort(scores, dim=-1).values
        logits = (
            -(
                (
                    scores_sorted.unsqueeze(-1)
                    @ all_ones.unsqueeze(-1).transpose(-1, -2)
                    - all_ones.unsqueeze(-1) @ scores.unsqueeze(-1).transpose(-1, -2)
                )
                ** 2
            )
            / self.temp
        )
        # perform a softmax on the last dimension of logits
        soft_permutations = torch.softmax(logits, dim=-1)
        log_probs = model.log_prob(
            batch,
            perm_mat=(permutations - soft_permutations).detach() +
            soft_permutations,
        )
        return -log_probs.mean()

    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0) -> torch.Tensor:
        permutations = self.sample_hard_permutations(
            batch.shape[0], unique_and_resample=True, gumbel_std=temperature
        ).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()


class PermutationMatrixLearningModule(PermutationLearningModule, abc.ABC):
    def __init__(
        self,
        in_features: int,
        set_gamma_uniform: bool = False,
        set_gamma_custom: Optional[List[List[int]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        if set_gamma_uniform:
            self.register_parameter(
                "_gamma", torch.nn.Parameter(torch.ones((in_features, in_features))))
        elif set_gamma_custom is not None:
            all_perm_mats = []
            for perm in set_gamma_custom:
                all_perm_mats.append(turn_into_matrix(
                    torch.IntTensor(perm)).float())
            all_perm_mats = torch.stack(all_perm_mats)
            self.register_parameter(
                "_gamma", torch.nn.Parameter(torch.mean(all_perm_mats, dim=0))
            )
        else:
            self.register_parameter(
                "_gamma", torch.nn.Parameter(
                    torch.randn(in_features, in_features))
            )

    def sample_hard_permutations(
        self, num_samples: int, return_noises: bool = False, unique_and_resample: bool = False, gumbel_std: float = 1.0
    ):
        if self.fixed:
            return turn_into_matrix(self.fixed_perm).unsqueeze(0).repeat(num_samples, 1, 1)
        else:
            gumbel_noise = sample_gumbel_noise(
                (num_samples, *self.gamma.shape),
                device=self.gamma.device,
                std=gumbel_std,
            )
            permutations = hungarian(
                self.gamma + gumbel_noise).to(self.gamma.device)
            if unique_and_resample:
                permutations = torch.unique(permutations, dim=0)
                permutations = permutations[
                    torch.randint(
                        0,
                        permutations.shape[0],
                        (num_samples,),
                        device=permutations.device,
                    )
                ]
            # turn permutations into permutation matrices
            ret = torch.stack([turn_into_matrix(perm)
                              for perm in permutations])
            # add some random noise to the permutation matrices
            if return_noises:
                return ret, gumbel_noise
            return ret

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        raise NotImplementedError

    def flow_learning_loss(self, model: OSlow, batch: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        permutations = self.sample_hard_permutations(
            batch.shape[0], unique_and_resample=True, gumbel_std=temperature
        ).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()


class SoftSinkhorn(PermutationMatrixLearningModule):
    def __init__(
        self, in_features: int, *args, temp: float = 0.1, iters: int = 20, **kwargs
    ):
        super().__init__(in_features, *args, **kwargs)
        self.temp = temp
        self.iters = iters

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        _, gumbel_noise = self.sample_hard_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        soft_permutations = _sinkhorn(
            self.gamma + gumbel_noise, iters=self.iters, temp=self.temp
        )
        log_probs = model.log_prob(batch, perm_mat=soft_permutations)
        return -log_probs.mean()


class GumbelSinkhornStraightThrough(PermutationMatrixLearningModule):
    def __init__(
        self,
        in_features: int,
        *args,
        temp: float = 0.1,
        iters: int = 20,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        self.temp = temp
        self.iters = iters

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_hard_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        soft_permutations = _sinkhorn(
            self.gamma + gumbel_noise, iters=self.iters, temp=self.temp
        )
        log_probs = model.log_prob(
            batch,
            perm_mat=(permutations - soft_permutations).detach() +
            soft_permutations,
        )
        return -log_probs.mean()


class PermutationMatrixLearningModuleWithBuffer(PermutationMatrixLearningModule):
    def __init__(
        self,
        in_features: int,
        num_samples: int,
        buffer_size: int,
        buffer_update: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        self.num_samples = num_samples
        self.buffer_size = buffer_size
        self.permutation_buffer = torch.zeros(
            buffer_size, in_features, in_features).to(self.gamma.device)
        self.permutation_buffer_scores = torch.full(
            (buffer_size,), float("-inf"), device=self.gamma.device)
        self.buffer_update = buffer_update

    def get_best(self, temperature: float = 1.0):
        # get the permutation corresponding to the max score
        if self.fixed:
            return super().get_best(temperature)
        idx_best = torch.argmax(self.permutation_buffer_scores).item()
        if self.permutation_buffer_scores[idx_best] == float("-inf"):
            return super().get_best(temperature)
        return self.permutation_buffer[idx_best]

    def _get_in_buffer_index(
        self,
        permutations: torch.Tensor,
    ):
        self.permutation_buffer = self.permutation_buffer.to(
            permutations.device)
        self.permutation_buffer_scores = self.permutation_buffer_scores.to(
            permutations.device)
        with torch.no_grad():
            # return a mask of size (permutations.shape[0], ) where the i-th element is the index of the i-th permutation in the buffer and -1 if it is not in the buffer
            msk = (self.permutation_buffer.reshape(self.permutation_buffer.shape[0], -1).unsqueeze(0).long(
            ) == permutations.reshape(permutations.shape[0], -1).unsqueeze(1).long()).all(dim=-1).long()
            # add a column of all ones to the beginning of msk
            msk = torch.cat(
                [torch.ones((msk.shape[0], 1), device=msk.device), 2 * msk], dim=1)
            idx = torch.argmax(msk, dim=-1) - 1
            # if self.permutation_buffer_score of the i-th permutation is -inf, then idx[i] = -1
            idx = torch.where(
                self.permutation_buffer_scores[idx] == float("-inf"),
                torch.full_like(idx, -1),
                idx,
            )
            return idx

    def update_buffer(
        self,
        dloader: torch.utils.data.DataLoader,
        model: OSlow,
        temperature: float = 1.0,
    ):
        with torch.no_grad():
            # (1) sample a set of "common" permutations
            new_permutations = self.sample_hard_permutations(
                self.num_samples * self.buffer_update, gumbel_std=temperature).detach()
            new_unique_perms, counts = torch.unique(
                new_permutations, dim=0, return_counts=True)
            # sort unique perms according to their counts and take the first buffer_size
            sorted_indices = torch.argsort(counts, descending=True)
            new_unique_perms = new_unique_perms[sorted_indices[:min(
                len(sorted_indices), self.buffer_size)]]

            # (2) go over the entire dataloader to compute the scores for this new set of common permutations
            new_scores = torch.zeros(
                len(new_unique_perms), device=self.gamma.device).float()
            _new_score_counts = torch.zeros(
                len(new_unique_perms), device=self.gamma.device).float()
            for x in dloader:
                x = x.to(self.gamma.device)
                L = 0
                R = 0
                for unique_perms_chunk in torch.split(new_unique_perms, self.num_samples):
                    R += unique_perms_chunk.shape[0]
                    new_unique_perms_repeated = unique_perms_chunk.repeat_interleave(
                        x.shape[0], dim=0)
                    x_repeated = x.repeat(len(unique_perms_chunk), 1)
                    log_probs = model.log_prob(
                        x_repeated, perm_mat=new_unique_perms_repeated).detach()
                    log_probs = log_probs.reshape(
                        len(unique_perms_chunk), x.shape[0])
                    new_scores[L:R] += log_probs.sum(dim=-1)
                    _new_score_counts[L:R] += x.shape[0]
                    L = R
            new_scores /= _new_score_counts

            # (3) update the buffer by first replacing the scores in the current buffer with the new scores
            # if the new_scores are better than what was there before
            idx = self._get_in_buffer_index(new_unique_perms)
            pos_idx = idx[idx >= 0]
            self.permutation_buffer_scores[pos_idx] = torch.where(
                new_scores[idx >= 0] > self.permutation_buffer_scores[pos_idx],
                new_scores[idx >= 0],
                self.permutation_buffer_scores[pos_idx],
            )

            # (4) for all the new permutations, add them to the buffer if their scores are
            # already better than the ones seen in the buffer
            if (idx == -1).any():
                # if it does, then we need to add new permutations to the buffer
                new_unique_perms = new_unique_perms[idx == -1]
                new_scores = new_scores[idx == -1]
                appended_permutations = torch.cat(
                    [self.permutation_buffer, new_unique_perms], dim=0)
                appended_scores = torch.cat(
                    [self.permutation_buffer_scores, new_scores], dim=0)

                # sort in a descending order according to the scores
                sorted_indices = torch.argsort(
                    appended_scores, descending=True)
                self.permutation_buffer = appended_permutations[sorted_indices[:self.buffer_size]]
                self.permutation_buffer_scores = appended_scores[sorted_indices[:self.buffer_size]]


class GumbelTopK(PermutationMatrixLearningModuleWithBuffer):

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations = self.sample_hard_permutations(
            self.num_samples, gumbel_std=temperature)
        unique_perms = torch.unique(permutations, dim=0)

        # shape: (num_uniques, ) -> calculate logits with Frobenius norm
        logits = torch.sum(
            unique_perms.reshape(
                unique_perms.shape[0], -1) * self.gamma.reshape(1, -1),
            dim=-1,
        )

        scores = torch.zeros(unique_perms.shape[0], device=self.gamma.device)
        # score[i] represents the log prob at permutation i
        idx = self._get_in_buffer_index(unique_perms)
        scores[idx >= 0] = self.permutation_buffer_scores[idx[idx >= 0]]

        if (idx == -1).any():
            unique_perms = unique_perms[idx == -1]
            b_size = batch.shape[0]
            n_unique = unique_perms.shape[0]

            unique_perms_repeated = unique_perms.repeat_interleave(
                b_size, dim=0)
            # shape: (batch * num_uniques, d)
            batch_repeated = batch.repeat(n_unique, 1)

            log_probs = model.log_prob(
                batch_repeated, perm_mat=unique_perms_repeated)
            log_probs = log_probs.reshape(
                n_unique, b_size
            )  # shape: (batch, num_uniques, )
            # shape: (num_uniques, )
            scores[idx == -1] = log_probs.mean(axis=-1)

        return - torch.softmax(logits, dim=0) @ scores


class ContrastiveDivergence(PermutationMatrixLearningModuleWithBuffer):

    def permutation_learning_loss(
        self, model: OSlow, batch: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        with torch.no_grad():
            permutations = self.sample_hard_permutations(
                self.num_samples, gumbel_std=temperature).detach()
            unique_perms, counts = torch.unique(
                permutations, dim=0, return_counts=True)

            scores = torch.zeros(
                unique_perms.shape[0], device=self.gamma.device)

            idx = self._get_in_buffer_index(unique_perms)
            scores[idx >= 0] = self.permutation_buffer_scores[idx[idx >= 0]]

            if (idx == -1).any():
                permutations_repeated = torch.repeat_interleave(
                    unique_perms[idx == -1], batch.shape[0], dim=0
                )
                batch_repeated = batch.repeat(len(unique_perms[idx == -1]), 1)

                all_log_probs = model.log_prob(
                    batch_repeated, perm_mat=permutations_repeated)
                scores[idx == -1] = all_log_probs.reshape(len(unique_perms[idx == -1]), -1).mean(dim=-1)

        all_energies = torch.einsum("ijk,jk->i", unique_perms, self.gamma)
        weight_free_term = torch.sum(all_energies * counts) / torch.sum(counts)
        return - torch.sum(scores * (all_energies - weight_free_term) * counts) / torch.sum(counts)
