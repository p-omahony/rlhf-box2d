import torch
import gym 
import numpy as np 

from pandas.core.frame import DataFrame
from typing import Dict

def compute_returns(rewards, discount_factor, normalize = True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

def compute_advantages(returns, values, normalize = True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages

def get_profit(info: Dict) -> float:

    return info["total_profit"]

def get_env_name(env: gym.Env) -> str:

    env_name = env.unwrapped.spec.id
    return env_name

def reset_env(env: gym.Env) -> np.ndarray: 
    env_name = get_env_name(env)

    if env_name in ("stocks-v0", "forex-v0"): 
        state = env.reset()
    else: 
        state, _ = env.reset()

    return state

def get_trading_env_args(df: DataFrame, train_prop: float, window_size: int) -> Dict: 

    n = len(df)
    train_size = int(n * train_prop)
    
    args = {
        "train": {
            "frame_bound": (window_size, train_size),
            "window_size": window_size
        }, 
        "test": {
            "frame_bound": (train_size+window_size+1, n),
            "window_size": window_size
        }
    }    
    return args