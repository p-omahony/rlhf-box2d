from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast

from datasets import HFLunarLander
from models.reward import RewardModel


def to_arr(arrays):
  arrays = arrays[1:-1].split('dtype=float32), ')
  final_array = []
  for arr in arrays:
    arr = arr.replace('],','])')
    arr = arr.replace('\n', '')
    arr= arr.replace('array(', '').replace(')', '')
    arr = arr.replace('dtype=float32', '')
    arr = ast.literal_eval(arr)
    final_array.append(np.array(arr))
  return final_array

def label_encode(x):
  if x==0:
    return [1,0]
  elif x==1:
    return [0,1]
  else:
    return [0.5,0.5]
  

print('Data processing...')
df = pd.read_csv('./data/logs_ll2.csv')
df['states'] = df['states'].apply(lambda x: to_arr(x))
df['actions'] = df['actions'].apply(lambda x: ast.literal_eval(x))
df['label'] = df['label'].apply(lambda x: label_encode(int(x)))
df['states1'] = df['states1'].apply(lambda x: to_arr(x))
df['actions1'] = df['actions1'].apply(lambda x: ast.literal_eval(x))
df['label1'] = df['label1'].apply(lambda x: label_encode(int(x)))

r_df = df[['states', 'actions', 'label']]
l_df = df[['states1', 'actions1', 'label1']]
l_df.columns = ['states', 'actions', 'label']
final_df = pd.concat([r_df, l_df])

dataset = HFLunarLander(final_df)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print('Dataset size:', len(dataset))

print('Loading model...')
model = RewardModel()
model.load_state_dict(torch.load('./weights/reward_model_0.5735118846098582_2.pth'))
print('Training...')
model.train()
optimizer = optim.Adam(model.parameters(), lr = 0.00001)
loss_func = nn.CrossEntropyLoss()

epochs = 25
pbar = tqdm(range(epochs))
history_loss = []
for epoch in pbar:
  running_loss = 0.0
  c=0
  for states, actions, label in dataloader:
    predicted_rewards, predicted_rewards1 = [], []
    optimizer.zero_grad()
    cr = torch.zeros(2)
    for k in range(len(states[0])):
      r = model(states[0][k], actions[0][k])
      cr = torch.add(cr, r)
    
    prob = F.softmax(cr)
    loss = loss_func(prob, label)
  
    loss.backward()
    optimizer.step()
    c+=1

    running_loss += loss.item()
  history_loss.append(running_loss/c)
  pbar.set_description(f'Loss: {running_loss/c}')


plt.plot(range(epochs), history_loss)
plt.savefig('reward_loss.png')

torch.save(model.state_dict(), f'./weights/reward_model_{history_loss[-1]}_3.pth')