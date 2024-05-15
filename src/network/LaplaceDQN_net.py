# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):

    def __init__(self, config):
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_sensitivities = config.num_sensitivities
        output_dim = self.action_dim * self.num_sensitivities

        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(self.input_dim, 24) # originally 128
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, output_dim)
        self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_sensitivities)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) 
        x1 = self.reshape_layer(x)
        
        x = torch.cumsum(x1, dim=2)
        x = torch.flip(x, dims=[2])
                
        diffs = torch.diff(x, dim=2)
        #print(x.shape, diffs.shape, x, diffs)
        # assert x == diffs
        assert torch.all(diffs <= 0), "Output Tensor is not non-increasing along dimension 2"
        
        return x