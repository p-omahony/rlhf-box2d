import gym 

import torch
import torch.nn.functional as F
import torch.distributions as distributions

from typing import Optional

import models.reinforce as rforce
import models.ppo as p
from utils.rl import compute_advantages, compute_returns


def train_one_episode(
    env: gym.Env, 
    policy, 
    optimizer, 
    discount_factor: float, 
    method: str, 
    max_actions: int,  
    ppo_steps: Optional[int]=None, 
    ppo_clip: Optional[int]=None
):
    """Train one episode of the environment using the parametrized policy.
    
    Args:
        env (gym.Env): The environment to train the policy on.
        policy (torch.nn.Module): The parametrized policy to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        discount_factor (float): The discount factor to use for computing returns.
        method (str): The method to use for training. Must be either "reinforce" or "ppo".
        max_actions (int): The maximum number of actions to take in the environment.
        ppo_steps (Optional[int]): The number of PPO steps to take. Defaults to None. Used if method is "ppo".
        ppo_clip (Optional[int]): The PPO clipping parameter. Defaults to None. Used if method is "ppo".
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, float]: The policy loss, value loss, and total reward obtained in the episode.
    """
    
    if method not in ("reinforce", "ppo"): 
        raise ValueError("method must be either 'reinforce' or 'ppo'.")
    
    if method == "ppo": 
        if ppo_steps is None or ppo_clip is None: 
            raise ValueError("ppo_steps and ppo_clip must be specified for PPO training.") 

    policy.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    episode_reward = 0

    state, _ = env.reset()
    terminated, c = False, 0
    
    while not terminated and c < max_actions: 
        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)
        
        if method == "ppo": 
            action_pred, value_pred = policy(state)  
            values.append(value_pred) 
        else:
            action_pred = policy(state)

        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        state, reward, terminated, truncated, info = env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        
        episode_reward += reward
        c+=1

    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)

    returns = compute_returns(rewards, discount_factor)

    if method == "ppo":        
        values = torch.cat(values).squeeze(-1)
        advantages = compute_advantages(returns, values)
        policy_loss, value_loss = p.update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    else: 
        policy_loss = rforce.update_policy(returns, log_prob_actions, optimizer)
        value_loss = None

    return policy_loss, value_loss, episode_reward