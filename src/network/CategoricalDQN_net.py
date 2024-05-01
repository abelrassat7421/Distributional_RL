# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CategoricalDQNNet(nn.Module):

    def __init__(self, config):
        super(CategoricalDQNNet, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_atoms = config.categorical_n_atoms
        output_dim = self.action_dim * self.num_atoms

        self.atoms = torch.linspace(
            float(config.categorical_Vmin),
            float(config.categorical_Vmax),
            config.categorical_n_atoms,
        )  # Z

        self.layer1 = nn.Linear(self.input_dim, 24)
        self.layer2 = nn.Linear(24, output_dim)
        self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_atoms)  # Lambda layer for reshaping

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.reshape_layer(x)
        x = F.softmax(x, dim=-1)

        action_values = torch.tensordot(x, self.atoms, dims=1)
        #print("DEBUG #2: Action values in NN evolution --:", action_values[0]) # TODO D1
        # Create an index of the max action value in each batch
        idx = torch.argmax(action_values, dim=1).to(torch.int64)
        idx = idx.view(x.shape[0], 1, 1).expand(-1, -1, self.num_atoms)

        # Adjust the index to: [[0, 1], [1, 0], [2, 1], [3, 1], etc.]
        # First number is row (batch) number, second number is the argmax index
        
        #idx = torch.stack((torch.arange(x.shape[0]), idx), dim=1)
        # Gather probability histogram for actions with max action values
        actorNet_output_argmax = torch.gather(x, dim=1, index=idx).squeeze(1)
        #print("DEBUG 7: actorNet_output_argmax.shape", actorNet_output_argmax.shape)
         
        # x: (batch_size, action_dim, num_atoms)
        # actorNet_output_argmax: (batch_size, num_atoms) -> atoms of action with max expected value 
        return x, actorNet_output_argmax