import torch
from torch import nn


class MemoryController(nn.Module):
    def __init__(self, cfg, input_dim = 3):
        super(MemoryController, self).__init__()
        self.future_nums = cfg['model_params']['future_num_frames']
        self.model_name = cfg['model_params']['model_name']
        self.layer = nn.Sequential(
            nn.Linear(in_features=self.future_nums * (2 * input_dim), out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=32),
            nn.Sigmoid(),
            nn.Linear(in_features=32, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=1),
            nn.Sigmoid()

        )

    def forward(self, x, x_hat):
        """
        params:
            x: bs, future * 2
            xf: bs, future * 2
        """
        x = torch.flatten(x, 1)
        x_hat = torch.flatten(x_hat, 1)
        x = torch.cat((x, x_hat), 1)
        x = torch.flatten(x, 1)
        x = self.layer(x)
        return x
