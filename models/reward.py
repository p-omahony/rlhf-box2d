import torch.nn as nn
import torch

class RewardModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.fc = nn.Sequential(
        nn.Linear(9,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128,2)
    )

  def forward(self, s, a):
    #out = self.conv(s)
    s = s.view(-1, 8)
    a = a.view(-1, 1)
    out = torch.concat((s, a), dim=1)
    out = self.fc(out)
    return out