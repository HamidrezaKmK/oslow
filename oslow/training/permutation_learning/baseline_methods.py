import torch
import abc

from oslow.models import OSlow
from oslow.training.utils import sample_gumbel_noise, hungarian, turn_into_matrix
from typing import Optional, Literal, List, Callable
from oslow.training.utils import sinkhorn as sinkhorn_operator

class PermutationLearningModule(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        in_features: int,
        initialization_function: Callable[[int], torch.Tensor],
    ):
        super().__init__()
        self.in_features = in_features
        self.gamma = torch.nn.Parameter(initialization_function(in_features))
        self.fixed_perm = None
        
    def freeze(
        self,
        perm: Optional[List[int]] = None,
    ):
        # freezes the gamma parameter
        # and if a permutation or a list of permutations is specified, 
        # after this all the samples obtained from the module
        # will coincide with one of the specified permutations
        # it would basically replace the permutation distribution to
        # a uniform distribution over the specified set of permutations
        if perm:
            self.fixed_perm = torch.IntTensor(perm)
        self.gamma.requires_grad = False

    def unfreeze(
        self,
    ):
        # It would allow the parameters to be trained again
        self.gamma.requires_grad = True
        self.fixed_perm = None
        

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        raise NotImplementedError

    def flow_learning_loss(self, model: OSlow, batch: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        The loss function used for training the `model` (which is a normalizing flow model) with the current permutation module.
        
        In a nutshell, it generates a set of permutations (and then resamples to make it uniform) 
        and uses the data in `batch` as input to the model to extract an estimate for the log probabilities of the permutations. 
        The loss is then the negative of the mean of these log probabilities.
        """
        permutations = self.sample_permutations(
            batch.shape[0], unique_and_resample=True, gumbel_std=temperature
        ).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()

    def get_best(self, temperature: float = 1.0, num_samples: int = 100):
        """
        Gets `num_samples` samples from the module and returns the sample that appears the most as the best permutation.
        """
        permutation_samples = self.sample_permutations(
            num_samples, unique_and_resample=False, gumbel_std=temperature).detach()
        # find the majority of permutations being sampled
        permutation, counts = torch.unique(
            permutation_samples, dim=0, return_counts=True
        )
        # find the permutation with the highest count
        return permutation[counts.argmax()]
    
    def sample_permutations(
        self, num_samples: int, return_noises: bool = False, unique_and_resample: bool = False, gumbel_std: float = 1.0
    ):
        """
        Sample `num_samples` permutations from the module.

        Args:
            num_samples (int): The number of permutations to sample.
            return_noises (bool, optional): Return the gumbel noises used for sampling. Defaults to False.
            unique_and_resample (bool, optional): 
                If set to true, the distribution of the permutations becomes uniform over the support of the originally sampled permutations. Defaults to False.
            gumbel_std (float, optional): Standard deviation of the gumbel noises used (can be interpreted as a temperature metric) Defaults to 1.0.

        Returns:
            torch.Tensor, (torch.Tensor, optional): 
                A tensor of shape (num_samples, in_features, in_features) which is a list of permutation matrices.
                The gumbel noises used for sampling is also returned if `return_noises` is set to True.
        """
        
        gumbel_noise = sample_gumbel_noise(
            (num_samples, *self.gamma.shape),
            device=self.gamma.device,
            std=gumbel_std,
        )
        
        if self.gamma.dim() == 1:
            
            # The sampling is based on the list parameterization of permutations
            # gumbel noise is added to the list and an argsort is performed to obtain a permutation
            
            if self.fixed_perm is not None:
                return self.fixed_perm.unsqueeze(0).repeat(num_samples, 1)
            else:
                scores = self.gamma + gumbel_noise
                # perform argsort on every line of scores
                permutations = torch.argsort(scores, dim=-1)

        elif self.gamma.dim() == 2:
            
            # The sampling is based on the permutation matrix parameterization
            # gumbel noise is added to the permutation matrix and the Hungarian algorithm
            # matching function is applied to obtain a permutation
            
            if self.fixed_perm is not None:
                return turn_into_matrix(self.fixed_perm).unsqueeze(0).repeat(num_samples, 1, 1)
            else:
                permutations = hungarian(
                    self.gamma + gumbel_noise).to(self.gamma.device)
        else:
            raise ValueError(
                f"Expected gamma to have 1 or 2 dimensions, got {self.gamma.dim()}"
            )
            
        # if unique_and_resample is set to True, then we need to sample unique permutations and resample them
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
        


class SoftSort(PermutationLearningModule):
    """
    Based on https://proceedings.mlr.press/v119/prillo20a/prillo20a.pdf
    titled "SoftSort: A Continuous Relaxation for the argsort Operator"
    """
    def __init__(
            self,
            in_features: int,
            *args,
            temp: float = 0.1,
            **kwargs
    ):
        super().__init__(in_features=in_features, *args, **kwargs)
        self.temp = temp
        
    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_permutations(
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

class SoftSinkhorn(PermutationLearningModule):
    """
    Directly feeds the output of the Sinkhorn operator to the flow model
    """
    def __init__(
        self, in_features: int, *args, temp: float = 0.1, iters: int = 20, **kwargs
    ):
        super().__init__(in_features, *args, **kwargs)
        self.temp = temp
        self.iters = iters

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        _, gumbel_noise = self.sample_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        soft_permutations = sinkhorn_operator(
            self.gamma + gumbel_noise, iters=self.iters, temp=self.temp
        )
        log_probs = model.log_prob(batch, perm_mat=soft_permutations)
        return -log_probs.mean()


class GumbelSinkhornStraightThrough(PermutationLearningModule):
    """
    Uses the straight-through gradient estimator to backpropagate through the Sinkhorn operator
    """
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
        
        permutations, gumbel_noise = self.sample_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        
        soft_permutations = sinkhorn_operator(
            self.gamma + gumbel_noise, iters=self.iters, temp=self.temp
        )
        log_probs = model.log_prob(
            batch,
            perm_mat=(permutations - soft_permutations).detach() +
            soft_permutations,
        )
        return -log_probs.mean()

