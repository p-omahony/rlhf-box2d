import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import gym
import numpy as np

from utils.rl import compute_advantages, compute_returns
from utils.functions import save_weights, load_config
from models.ppo import update_policy, MultiLayerPerceptron, ActorCritic

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

train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

cfg = load_config('./config.yaml')

input_dim = train_env.observation_space.shape[0]
hidden_dim = cfg['ppo']['actor_critic']['hidden_dim']
output_dim = train_env.action_space.n

actor = MultiLayerPerceptron(input_dim, hidden_dim, output_dim)
critic = MultiLayerPerceptron(input_dim, hidden_dim, 1)

ppo = ActorCritic(actor, critic)

epsilon = cfg['ppo']['hyperparameters']['epsilon']
steps = cfg['ppo']['hyperparameters']['steps']
n_exps = cfg['ppo']['hyperparameters']['n_exps']
print_freq = cfg['ppo']['hyperparameters']['print_freq']
reward_threshold = cfg['ppo']['hyperparameters']['reward_threshold']
gamma = cfg['ppo']['hyperparameters']['gamma']
lr = cfg['ppo']['hyperparameters']['lr']
episodes = cfg['ppo']['hyperparameters']['episodes']
max_actions = cfg['ppo']['hyperparameters']['max_actions']

optimizer = optim.Adam(ppo.parameters(), lr = lr)

train_rewards = []
test_rewards = []

for episode in range(1, episodes+1):
    
    clip_loss, value_loss, train_reward = train_one_episode(train_env, ppo, optimizer, gamma, steps, epsilon, max_actions)
    test_reward = evaluate_one_episode(test_env, ppo, max_actions)
    
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    #get the mean rewards of the last n_exps 'experiments'    
    mean_train_rewards = np.mean(train_rewards[-n_exps:])
    mean_test_rewards = np.mean(test_rewards[-n_exps:])
    
    if episode % print_freq == 0:
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
    
    if mean_test_rewards >= reward_threshold:
        print(f'Reached reward {str(reward_threshold)} in {episode} episodes')
        save_weights(ppo, f'ppo_{reward_threshold}_{episode}.pt')
        break