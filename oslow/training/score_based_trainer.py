"""
Script relating to novel permutation learning techniques that was used in the paper.
"""

from .trainer import Trainer
import torch
import wandb
from .permutation_learning.buffered_methods import BufferedPermutationLearning

class ScoreBasedTrainer(Trainer):
    # This trainer incorporates contrastive divergence and the energy-based permutation learning method 
    # that were used in the paper.
    
    # It is based on assinging a score to each permutation which is the average log_prob of the model given
    # that ordering. The score is then used to train the permutation model to only sample permutations
    # with the highest scores.
    
    def __init__(
        self,
        *args,
        perform_final_buffer_search: bool = False,
        **kwargs,
    ):
        """
        Everything is the same as the Trainer class except for the perform_final_buffer_search parameter.
        
        perform_final_buffer_search: bool
            If True, the final phase of the training will be performed. This phase involves training the
            permutation learning model with the best permutations from the buffer. And the trainings are
            done sequentially with OSLow being instantiated as a new fixed permutation flow.
        """
        super().__init__(*args, **kwargs)
        
        # We know that the permutation learning module is a BufferedPermutationLearning object
        if not isinstance(self.permutation_learning_module, BufferedPermutationLearning):
            raise ValueError(
                "The permutation learning module must be a BufferedPermutationLearning object for this trainer to work!"
            )
            
        self.permutation_learning_module: BufferedPermutationLearning
         
        self.perform_final_buffer_search = perform_final_buffer_search
        if self.perform_final_buffer_search and not hasattr(self.permutation_learning_module, "update_buffer"):
            raise ValueError(
                "The permutation learning module must have a buffer to perform final buffer search"
            )
        
        
    def final_phase(self):
            
        final_phase_buffer_size = len(
            self.permutation_learning_module.permutation_buffer)
        final_phase_epoch_count = self.max_epochs * \
            (self.permutation_frequency +
             self.flow_frequency) // final_phase_buffer_size
        if final_phase_epoch_count > 0 and final_phase_buffer_size > 0 and hasattr(self.permutation_learning_module, "update_buffer"):
            cap = min(10, len(
                self.permutation_learning_module.permutation_buffer))
            candidate_permutations = self.permutation_learning_module.permutation_buffer[:cap].cpu(
            ).numpy()
            best_avg_loss = None
            best_perm = None

            def tmp_fn(perm_list, best_avg_loss, best_perm, lbl):
                self.permutation_learning_module.freeze(perm_list)
                self.model.reinitialize()
                self.flow_optimizer = self.flow_optimizer_fn(
                    self.model.parameters())
                for _ in range(final_phase_epoch_count):
                    avg_loss = self.flow_train_step(
                        temperature=0.0, lbl=f"{lbl}/{'-'.join(map(str, perm_list))}-")
                if best_avg_loss is None or avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    best_perm = perm_list
                return best_avg_loss, best_perm

            for perm in candidate_permutations:
                perm_list = perm.argmax(-1).tolist()
                best_avg_loss, best_perm = tmp_fn(
                    perm_list, best_avg_loss, best_perm, lbl="buffered_flows")

            _, _ = tmp_fn(best_perm, best_avg_loss,
                          best_perm, lbl="final_flow")

    def learn_permutation(self, epoch: int):
        # Update the buffer with the current model before training the permutation learning model
        self.permutation_learning_module.update_buffer(
            dloader=self.perm_dataloader,
            model=self.model,
            temperature=self.get_temperature(epoch),
        )
        
        # For permutation_frequency number of steps, train the permutation learning model
        # by getting the loss and then performing a backward pass
        for _ in range(self.permutation_frequency):
            loss = self.permutation_learning_module.permutation_learning_loss(
                dloader=self.perm_dataloader,
                model=self.model,
                temperature=self.get_temperature(epoch),
            )
            loss.backward()
            self.permutation_optimizer.step()
            self.perm_step_count += 1
            wandb.log({"permutation/step": self.perm_step_count})
            wandb.log({"permutation/loss": loss.item()})
    
            
            if isinstance(
                self.permutation_scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.permutation_scheduler.step(loss.item())
            else:
                self.permutation_scheduler.step()
                
    