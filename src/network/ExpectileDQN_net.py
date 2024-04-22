# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpectileDQNNet(nn.Module):

    def __init__(self, config):
        super(ExpectileDQNNet, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim
        self.num_expectiles = config.num_expectiles
        self.output_dim = self.action_dim * self.num_expectiles 
        self.expectile_mean_idx = int(config.num_expectiles / 2)

        self.BATCH_SIZE = self.config.BATCH_SIZE

        # note that middle expectile statistic is in fact the mean, i.e. tau_{middle}
        self.cum_density = torch.linspace(0.01, 0.99, self.num_expectiles) 

        self.layer1 = nn.Linear(self.input_dim, 24)
        self.layer2 = nn.Linear(24, self.action_dim)
        self.reshape_layer = lambda x: x.view(-1, self.action_dim, self.num_expectiles)  # Lambda layer for reshaping


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.reshape_layer(x)
        
        # get the action values <=> mid expectile
        # tf.cast is to cast the action values to int32
        action_values = x[:, :, self.expectile_mean_idx]
        action = torch.argmax(action_values, axis=1).to(torch.int64)
        
        # to get the optimal action expectiles
        # size = [batch_size, 2 actions, expectiles per action]
        # we need to generate the correct index, size = [batch_size, argmax index]
        idx = torch.stack((torch.arange(x.shape[0]), action), dim=1)

        # the final result is a [batch_size, expectiles] tensor for optimal actions
        # May require some Debgging as for CatgericalDQN
        optimal_action_expectiles = torch.gather(x, dim=1, index=idx).squeeze(1)

        return x, optimal_action_expectiles
    

# Might need to define the loss function as a extension of the torch.autograd.Function
# to be sure that gradients can be handled manually
def expectile_regression_loss(y_true, y_predict):
    """
    The loss function that is passed to the network
    :param y_true: True label, distribution after imputation (batch_size, number of z values)
    :param y_predict: predicted label, expectile_predict (batch_size, number of expectiles)
    :return: expectile loss between the target expectile and the predicted expectile
    """
    cum_density = torch.linspace(0.01, 0.99, y_predict.shape[1])   

    batch_loss = []
    for i in range(y_predict.shape[0]):
        expectile_predict = y_predict[i]
        z = y_true[i]
        loss_val = 0

        for k in range(y_predict.shape[1]):
            diff = z - expectile_predict[k]
            diff_square = diff.pow(2)

            er_loss = torch.mean(torch.where(diff > 0,
                                                cum_density[k] * diff_square,  # tau * (Z-q)^2
                                                (1 - cum_density[k]) * diff_square))  # (1-tau) * (Z-q)^2
            # sum over all k statistics
            loss_val += er_loss

        # get batch loss size=(1, 32)
        batch_loss.append(loss_val)

    return torch.mean(batch_loss)