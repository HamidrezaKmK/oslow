from .baseline_methods import PermutationLearningModule
import torch
from oslow.models import OSlow
from oslow.training.utils import sample_gumbel_noise, hungarian, turn_into_matrix

class BufferedPermutationLearning(PermutationLearningModule):
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
        if self.fixed_perm is not None:
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
            new_permutations = self.sample_permutations(
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

    def permutation_learning_loss(
        self, 
        dloader: torch.utils.data.DataLoader, 
        model: OSlow, 
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # The permutation learning loss in these scenarios take in an entire 
        # dataloader and are therefore different from other methods
        raise NotImplementedError

class GumbelTopK(BufferedPermutationLearning):

    def permutation_learning_loss(
        self, 
        dloader: torch.utils.data.DataLoader, 
        model: OSlow, 
        temperature: float = 1.0,
    ) -> torch.Tensor:
        permutations = self.sample_permutations(
            self.num_samples, gumbel_std=temperature)
        unique_perms = torch.unique(permutations, dim=0)

        # shape: (num_uniques, ) -> calculate logits with Frobenius norm
        logits = torch.sum(
            unique_perms.reshape(
                unique_perms.shape[0], -1) * self.gamma.reshape(1, -1),
            dim=-1,
        )
        
        with torch.no_grad():
            scores = torch.zeros(unique_perms.shape[0], device=self.gamma.device)
            # score[i] represents the log prob at permutation i
            idx = self._get_in_buffer_index(unique_perms)
            scores[idx >= 0] = self.permutation_buffer_scores[idx[idx >= 0]]

            if (idx == -1).any():
                unique_perms = unique_perms[idx == -1]
                batch_cnt = 0
                for batch in dloader:
                    batch = batch.to(self.gamma.device)
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
                    scores[idx == -1] += log_probs.mean(axis=-1)
                    batch_cnt += b_size
                scores[idx == -1] /= batch_cnt

        return - torch.softmax(logits, dim=0) @ scores


class ContrastiveDivergence(BufferedPermutationLearning):

    def permutation_learning_loss(
        self, 
        model: OSlow, 
        dloader: torch.utils.data.DataLoader,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        permutations = self.sample_permutations(
            self.num_samples, gumbel_std=temperature).detach()
        unique_perms, counts = torch.unique(
            permutations, dim=0, return_counts=True)

        with torch.no_grad():
            # Compute the scores for every permutation
            # the score for every permutation is the average log prob you will get using the permutation
            # for this step, we don't need the gradients
            scores = torch.zeros(
                unique_perms.shape[0], device=self.gamma.device)

            idx = self._get_in_buffer_index(unique_perms)
            scores[idx >= 0] = self.permutation_buffer_scores[idx[idx >= 0]]

            if (idx == -1).any():
                batch_cnt = 0
                for batch in dloader:
                    batch = batch.to(self.gamma.device)
                    permutations_repeated = torch.repeat_interleave(
                        unique_perms[idx == -1], batch.shape[0], dim=0
                    )
                    batch_repeated = batch.repeat(len(unique_perms[idx == -1]), 1)

                    all_log_probs = model.log_prob(
                        batch_repeated, perm_mat=permutations_repeated)
                    scores[idx == -1] += all_log_probs.reshape(len(unique_perms[idx == -1]), -1).mean(dim=-1)
                    batch_cnt += batch.shape[0]
                scores[idx == -1] /= batch_cnt
                

        all_energies = torch.einsum("ijk,jk->i", unique_perms, self.gamma)
        weight_free_term = torch.sum(all_energies * counts) / torch.sum(counts)
        return - torch.sum(scores * (all_energies - weight_free_term) * counts) / torch.sum(counts)
