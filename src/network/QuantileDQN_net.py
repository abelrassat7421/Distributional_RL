# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantileDQNNet(nn.Module):

    def __init__(self, config):
        super(QuantileDQNNet, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_quantiles = config.num_quantiles
        output_dim = self.action_dim * self.num_quantiles

        self.k = config.huber_loss_threshold
        # quantiles, e.g. [0.125, 0.375, 0.625, 0.875]
        self.cum_density = (2 * np.arange(self.num_quantiles) + 1) / (2.0 * self.num_quantiles)
        # append r_0 = 0, [0.   , 0.125, 0.375, 0.625, 0.875]
        temp = np.sort(np.append(self.cum_density, 0))
        # calculate r_hat [0.0625, 0.25, 0.5, 0.75], see lemma 2 in the original paper.
        self.r_hat = np.array([(temp[j] + temp[j - 1]) / 2 for j in range(1, temp.shape[0])])

        self.layer1 = nn.Linear(self.input_dim, 24)
        self.layer2 = nn.Linear(24, output_dim)
        self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_quantiles)  # Lambda layer for reshaping

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.reshape_layer(x)

        # Get action values
        action_values = torch.sum(x, dim=2)
        idx = torch.argmax(action_values, dim=1).to(torch.int64)

        # Get indices for the optimal action quantiles
        idx = idx.view(x.shape[0], 1, 1).expand(-1, -1, self.num_quantiles)
        # idx = torch.stack([torch.arange(action.size(0)), action], dim=1) # This is the version from direct translation
        
        # Extract optimal action quantiles
        optimal_action_quantiles = torch.gather(x, dim=1, index=idx).squeeze(1)
 
        return x, optimal_action_quantiles
    

def huber_loss(mu, k):
    """
    equation 10 of the original paper
    :return:
    """
    #print("INSPECTING #1: ", mu)
    return torch.where(
        torch.abs(mu) < k,
        0.5 * torch.square(mu),
        k * (torch.abs(mu) - 0.5 * k)
    )
    

def quantile_huber_loss(y_true, y_predict, k):
    """
    The loss function that is passed to the network
    see algorithm 1 in the original paper for more details
    :param y_true: true label, quantiles_next, [batch_size, num_quantiles]
    :param y_predict: predicted label, quantiles, [batch_size, num_quantiles]
    :return: quantile huber loss between the target quantiles and the quantiles
    """
    # compute r_hat
    num_quantiles = y_true.shape[1]
    batch_size = y_true.shape[0]
    # taus
    cum_density = (2 * np.arange(num_quantiles) + 1) / (2.0 * num_quantiles)
    # append r_0 = 0, [0.  , 0.125, 0.375, 0.625, 0.875]
    temp = np.sort(np.append(cum_density, 0))
    # calculate r_hat [0.0625, 0.25, 0.5, 0.75], see lemma 2 in the original paper.
    r_hat = np.array([(temp[j] + temp[j - 1]) / 2 for j in range(1, temp.shape[0])])

    batch_loss = torch.zeros(batch_size)
    for each_batch in range(batch_size):
        each_transition_sample_loss = 0
        for i in range(y_true.shape[1]):
            diff = y_true[each_batch] - y_predict[each_batch, i]

            # calculate the expected value over j
            target_loss = torch.mean(
                (huber_loss(diff, k) *
                    torch.abs(r_hat[i] - (diff < 0).to(dtype=torch.float32))))

            # sum over i in algorithm 1
            each_transition_sample_loss += target_loss

        # get batch loss size=(32, 1)
        batch_loss[each_batch] = each_transition_sample_loss
    
    return torch.mean(batch_loss)


    