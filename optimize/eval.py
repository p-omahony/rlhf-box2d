import gym 

import torch
import torch.nn.functional as F

from typing import Union


def evaluate_one_episode(env: gym.Env, policy, method: str, max_actions: int, human_preferences='false', reward_model=None) -> Union[int, float]:
    """Evaluate one episode of the environment using the policy.
    
    Args:
        env (gym.Env): The environment to evaluate the policy on.
        policy (torch.nn.Module): The policy to evaluate.
        method (str): The method to use for evaluation. Must be either "reinforce" or "ppo".
        max_actions (int): The maximum number of actions to take in the environment.
        
    Returns:
        Union[int, float]: The total reward obtained in the episode."""
    
    if method not in ("reinforce", "ppo"): 
        raise ValueError("method must be either 'reinforce' or 'ppo'.")

    policy.eval()
    
    terminated, c = False, 0
    episode_reward = 0

    state, _ = env.reset()

    while not terminated and c < max_actions:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():

            if method == "ppo":
                action_pred, _ = policy(state)
            else:
                action_pred = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)
        state, reward, terminated, truncated, info = env.step(action.item())
        episode_reward += reward

        c+=1
        
    return episode_reward