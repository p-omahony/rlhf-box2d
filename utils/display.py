import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
from typing import List
import os
import numpy as np
from IPython import display
from skvideo.io import FFmpegWriter


def notebook_display_random_episode(env: gym.Env) -> None:
    env.reset()
    scene = plt.imshow(env.render(mode='rgb_array'))
    terminated = False
    while not terminated:
        scene.set_data(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = env.action_space.sample()
        state, reward, terminated, _ = env.step(action)

def write_video(outputfile: str, frames: List[np.array]) -> None: 
    base_path = './videos'
    path = os.path.join(base_path, outputfile)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    writer = FFmpegWriter(path, outputdict={
        '-vcodec': 'libx264', '-b': '300000000'
        })
    for i in range(len(frames)):
        writer.writeFrame(frames[i])
    writer.close()

def write_random_episode_video(env: gym.Env) -> None:
    env.reset()
    scene = env.render()
    terminated = False
    frames = [scene]
    while not terminated:
        frames.append(env.render())
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
    write_video('video.mp4', frames)

def write_episode(env: gym.Env, policy) -> None:
    state, _ = env.reset()
    scene = env.render()
    terminated = False
    frames = [scene]
    while not terminated:
        frames.append(env.render())
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)
        action = torch.argmax(action_prob, dim = -1)
        state, reward, terminated, truncated, info = env.step(action.item())
    write_video('solved.mp4', frames)

if __name__ == '__main__' :
    env = gym.make("LunarLander-v2", render_mode='rgb_array')
    write_random_episode_video(env)