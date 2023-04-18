import gym 

import torch
import torch.nn.functional as F

from typing import Union


def evaluate_one_episode(env: gym.Env, policy, max_actions: int) -> Union[int, float]:
    """Evaluate one episode of the environment using the policy.
    
    Args:
        env (gym.Env): The environment to evaluate the policy on.
        policy (torch.nn.Module): The policy to evaluate.
        max_actions (int): The maximum number of actions to take in the environment.
        
    Returns:
        Union[int, float]: The total reward obtained in the episode."""

    policy.eval()
    
    terminated, c = False, 0
    episode_reward = 0

    state, _ = env.reset()

    while not terminated and c < max_actions:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)
        state, reward, terminated, truncated, info = env.step(action.item())
        episode_reward += reward

        c+=1
        
    return episode_reward