import torch
import abc

from oslow.models import OSlow
from oslow.training.utils import sample_gumbel_noise, hungarian, listperm2matperm
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
                return listperm2matperm(self.fixed_perm, device=self.gamma.device).unsqueeze(0).repeat(num_samples, 1, 1)
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
        ret = torch.stack([listperm2matperm(perm, device=self.gamma.device)
                        for perm in permutations])
        # add some random noise to the permutation matrices
        if return_noises:
            return ret, gumbel_noise
        return ret
     
