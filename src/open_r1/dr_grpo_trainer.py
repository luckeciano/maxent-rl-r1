from open_r1.grpo_entropy_trainer import GRPOEntropyTrainer

class DrGRPOTrainer(GRPOEntropyTrainer):
    """
    DrGRPO trainer class.
    """
    
    def _compute_advantages(self, rewards, mean_grouped_rewards, std_grouped_rewards): 
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) 
        return advantages

    def _compute_final_loss(self, per_token_loss, completion_mask):
        loss = (per_token_loss * completion_mask).sum() # No division by the number of tokens
        return loss
    