# rlhf-box2d

## Installation    
You need to install `swig` and `ffmpeg`. Then run `pip install -r requirements.txt`.   
     
## Run   
### Train PPO
`python train_ppo.py`. Args :      
- `-- type`: either `exp` if you want to run a wandb sweep experiment, or `normal` to run with a specific configuration        
- `--config`: specify the path of the config file for ppo      
- `--env`: either `lunarlander` or `cartpole` (gym environments)
- `--human_preferences`: either `true` or `false` if you want to integrate human preferences       
- `--reward_ckpt`: specify the reward model checkpoint file if `--human_preferences` is `true`

### Train REINFORCE
`python train_reinforce.py`. Args :      
- `-- type`: either `exp` if you want to run a wandb sweep experiment, or `normal` to run with a specific configuration        
- `--config`: specify the path of the config file for ppo      
- `--env`: either `lunarlander` or `cartpole` (gym environments)

### Train the reward model
`python train_reward.py`.       

## References
- PPO Paper: https://arxiv.org/abs/1707.06347       
- Intro RL: https://lilianweng.github.io/posts/2018-02-19-rl-overview/     
- Policy gradients algorithms (with PPO): https://lilianweng.github.io/posts/2018-04-08-policy-gradient/         
