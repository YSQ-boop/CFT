import torch


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """
  

    return {
        "rewards": torch.float(0.0),  # Rewards for advantage calculation
        "scores": torch.float(0.0),  # Scores for dynamic filtering (0-1 reward)
        "extra_logs": {"dummy_scores": torch.float(0.0)},  # Additional logging info for wandb
    }
