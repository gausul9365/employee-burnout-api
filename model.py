import torch.nn as nn

class BurnoutModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.network = nn.Sequential(
      nn.Linear(12, 16),
      nn.ReLU(),

      nn.Linear(16, 8),
      nn.ReLU(),

      nn.Linear(8, 1),
      nn.Sigmoid()
    )


  def forward(self, x):
      return self.network(x)
  
