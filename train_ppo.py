import torch
import torch.nn.functional as F
import torch.distributions as distributions

from typing import Optional

from utils.rl import (
    compute_advantages, 
    compute_returns, 
    get_profit, 
    get_env_name, 
    reset_env
) 
from models.ppo import update_policy

def train_one_episode(
    env, 
    ppo, 
    optimizer, 
    discount_factor, 
    ppo_steps, 
    ppo_clip, 
    max_actions, 
    device: Optional[torch.device]=None):
    env_name = get_env_name(env)

    ppo.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    episode_reward = 0

    state = reset_env(env)
    terminated, c = False, 0

    while not terminated and c < max_actions: 
        
        state = torch.FloatTensor(state).unsqueeze(0)

        if device is not None:
            state = state.to(device)

        states.append(state)

        action_pred, value_pred = ppo(state) 
        action_prob = F.softmax(action_pred, dim = -1)
                
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        step = env.step(action.item())

        if len(step) == 4:
            state, reward, terminated, info = step
        elif len(step) == 5:
            state, reward, terminated, truncated, info = step
        else: 
            raise ValueError("Too many values returned from env.step().")
        
        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
        c+=1

    if env_name in ("stocks-v0", "forex-v0"): 
        profit = get_profit(info)          

    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = compute_returns(rewards, discount_factor)
    advantages = compute_advantages(returns, values)

    policy_loss, value_loss = update_policy(ppo, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    if env_name in ("stocks-v0", "forex-v0"): 
        return policy_loss, value_loss, episode_reward, profit

    return policy_loss, value_loss, episode_reward

def evaluate_one_episode(env, ppo, max_actions, device: Optional[torch.device]=None):
    env_name = get_env_name(env)

    ppo.eval()
    
    terminated, c = False, 0
    episode_reward = 0

    state = reset_env(env)

    while not terminated and c < max_actions:
        
        state = torch.FloatTensor(state).unsqueeze(0)
        if device is not None:
            state = state.to(device)

        with torch.no_grad():
            action_pred, _ = ppo(state)
            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)
        step = env.step(action.item())

        if len(step) == 4:
            state, reward, terminated, info = step
        elif len(step) == 5:
            state, reward, terminated, truncated, info = step
        else: 
            raise ValueError("Too many values returned from env.step().")

        episode_reward += reward
        c+=1 

    if env_name in ("stocks-v0", "forex-v0"): 
        episode_profit = get_profit(info)
        return episode_reward, episode_profit
    
    return episode_reward