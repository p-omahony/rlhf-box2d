import torch 

def update_policy(returns: torch.Tensor, log_prob_actions: torch.Tensor, optimizer) -> float:
    """Update policy using REINFORCE algorithm.
    
    Args:
        returns (torch.Tensor): The returns for each timestep.
        log_prob_actions (torch.Tensor): The log probability of the actions taken.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        
    Returns:
        float: loss value."""
    
    returns = returns.detach()
    loss = - (returns * log_prob_actions).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()