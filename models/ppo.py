import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as distributions

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, x):
        action_pred = self.actor(x)
        value_pred = self.critic(x)

        return action_pred, value_pred
    
    
def update_policy(ppo, states, actions, log_pi_theta_old, advantages, returns, optimizer, ppo_steps, epsilon):
    total_policy_loss = 0 
    total_value_loss = 0

    states = states.detach()
    actions = actions.detach()
    log_pi_theta_old = log_pi_theta_old.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    
    for _ in range(ppo_steps):
        #get new log prob of actions for all input states
        action_pred, value_pred = ppo(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        log_pi_theta = dist.log_prob(actions)
        #probability ratio (we have the log prob actions so we take the exp of the difference to compute the ratio)
        r_t_theta = (log_pi_theta - log_pi_theta_old).exp()
                
        cpi_loss = r_t_theta * advantages
        clip_value = torch.clamp(r_t_theta, min = 1.0 - epsilon, max = 1.0 + epsilon) * advantages
        
        clip_loss = - torch.min(cpi_loss, clip_value).mean()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).mean()

        optimizer.zero_grad()

        clip_loss.backward()
        value_loss.backward()

        optimizer.step()

        total_policy_loss += clip_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
