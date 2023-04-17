import torch
import torch.nn.functional as F
import torch.distributions as distributions

from utils.rl import compute_advantages, compute_returns
from models.ppo import update_policy

def train_one_episode(env, ppo, optimizer, discount_factor, ppo_steps, ppo_clip, max_actions):
    ppo.train()
        
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
        action_pred, value_pred = ppo(state)    
        action_prob = F.softmax(action_pred, dim = -1)
                
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, terminated, truncated, info = env.step(action.item())
        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
        c+=1

    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = compute_returns(rewards, discount_factor)
    advantages = compute_advantages(returns, values)

    policy_loss, value_loss = update_policy(ppo, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward

def evaluate_one_episode(env, ppo, max_actions):
    ppo.eval()
    
    terminated, c = False, 0
    episode_reward = 0

    state, _ = env.reset()

    while not terminated and c < max_actions:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = ppo(state)
            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)
        state, reward, terminated, truncated, info = env.step(action.item())
        episode_reward += reward

        c+=1
        
    return episode_reward