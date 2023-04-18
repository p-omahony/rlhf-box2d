import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import gym
import numpy as np
import wandb
import argparse

from utils.rl import compute_advantages, compute_returns
from utils.functions import save_weights, load_config
from models.ppo import update_policy, ActorCritic
from models.base_models import MultiLayerPerceptron

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

def main():
    if args.type == 'exp':
        run = wandb.init()

    if args.env == 'lunarlander':
        env_name = 'LunarLander-v2'
    train_env = gym.make(env_name)
    test_env = gym.make(env_name)
        
    cfg = load_config('./config-ppo.yaml')
    actor_cfg = cfg['ppo'][args.env]['actor']
    critic_cfg = cfg['ppo'][args.env]['critic']

    #this block (wandb experiment part) only works if and only if actor and critic have the same architecture
    if args.type == 'exp':
        actor_cfg['input_layer'][0][2] = wandb.config.hidden_dim
        critic_cfg['input_layer'][0][2] = wandb.config.hidden_dim
        for layer in actor_cfg['hidden_layers']:
            layer_name = layer[0]
            if layer_name == 'Linear':
                layer[1] = wandb.config.hidden_dim
                layer[2] = wandb.config.hidden_dim
        for layer in critic_cfg['hidden_layers']:
            layer_name = layer[0]
            if layer_name == 'Linear':
                layer[1] = wandb.config.hidden_dim
                layer[2] = wandb.config.hidden_dim
        actor_cfg['input_layer'][0][1] = wandb.config.hidden_dim
        critic_cfg['input_layer'][0][1] = wandb.config.hidden_dim

    actor = MultiLayerPerceptron(actor_cfg)
    critic = MultiLayerPerceptron(critic_cfg)

    ppo = ActorCritic(actor, critic)

    if args.type=='exp':
        lr = wandb.config.lr
        episodes = wandb.config.episodes
        max_actions = wandb.config.max_actions
    else:
        lr = cfg['ppo'][args.env]['hyperparameters']['lr']
        episodes = cfg['ppo'][args.env]['hyperparameters']['episodes']
        max_actions = cfg['ppo'][args.env]['hyperparameters']['max_actions']

    epsilon = cfg['ppo'][args.env]['hyperparameters']['epsilon']
    steps = cfg['ppo'][args.env]['hyperparameters']['steps']
    n_exps = cfg['ppo'][args.env]['hyperparameters']['n_exps']
    print_freq = cfg['ppo'][args.env]['hyperparameters']['print_freq']
    reward_threshold = cfg['ppo'][args.env]['hyperparameters']['reward_threshold']
    gamma = cfg['ppo'][args.env]['hyperparameters']['gamma']

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

        if args.type == 'exp':
            wandb.log({
                'episode': episode, 
                'mean_train_rewards': mean_train_rewards,
                'mean_test_rewards': mean_test_rewards,
                'clip_loss': clip_loss,
                'value_loss': value_loss
            })

        
        if episode % print_freq == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
        
        if mean_test_rewards >= reward_threshold:
            print(f'Reached reward {str(reward_threshold)} in {episode} episodes')
            save_weights(ppo, f'ppo_{reward_threshold}_{episode}.pt')
            
        if mean_test_rewards >= 200:
            break


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type')
parser.add_argument('-e', '--env', default='lunarlander')
args = parser.parse_args()
if args.type == 'exp':
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'mean_test_rewards'},
        'parameters': 
        {
            'hidden_dim': {'values': [128, 256, 512]},
            'episodes': {'values': [1000, 2000, 3000]},
            'lr': {'max': 0.001, 'min': 0.0001},
            'max_actions': {'values': [100, 500, 1000]}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='rlhf-box2d')
    wandb.agent(sweep_id, function=main, count=6)
else:
    main()