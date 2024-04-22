# -*- coding:utf-8 -*-
from src.utils.replay_memory import ReplayMemory, Transition
from src.utils.epsilon_greedy import select_action
from src.utils.visualization import plot_durations
from scipy.optimize import minimize, root
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from src.network.ExpectileDQN_net import ExpectileDQNNet, expectile_regression_loss
import random
from itertools import count
import math
import torch

class ExpectileDQNAgent:

    def __init__(self, config):
        self.config = config 
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.num_expectiles = config.num_expectiles
        self.num_imputed_samples = config.num_imputed_samples
        self.expectile_mean_idx = int(config.num_expectiles / 2)

        self.total_steps = 0
        self.num_episodes = config.num_episodes
        self.steps = config.steps
        self.BATCH_SIZE = config.BATCH_SIZE
        self.GAMMA = config.GAMMA
        self.LR = config.LR
        self.TAU = config.TAU
        self.device = config.device

        self.env = None
        # copying weights of base_net to policy_net and target_net
        self.policy_net = ExpectileDQNNet(config)
        self.target_net = ExpectileDQNNet(config)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.criterion = expectile_regression_loss

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_memory = ReplayMemory(self.replay_buffer_size)

        # self.keras_check = config.keras_checkpoint
        self.episode_durations = []
        self.check_model_improved = torch.tensor([0])
        self.best_max = torch.tensor([0])
        # note that tau_6 = 0.5 and thus this expectile statistic is in fact the mean
        # tau NOTE (because 11 expectiles are used as in the original paper)  
        self.cum_density = np.linspace(0.01, 0.99, config.num_expectiles)
        self.imputation_method = config.imputation_method

        # for select action (epsilon-greedy)
        self.steps_done = 0


    def transition(self):
        """
        In transition, the agent simply plays and records
        [current_state, action, reward, next_state, done]
        in the replay_memory

        Updating the weights of the neural network happens
        every single time the replay buffer size is reached.

        done: boolean, whether the game has ended or not.
        """
        for i_episode in range(self.num_episodes):
            state, info = self.env.reset() 
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            print('Episode: {} Reward: {} Max_Reward: {}'.format(i_episode, self.check_model_improved[0].item(), self.best_max[0].item()))
            print('-' * 64)
            self.check_model_improved = 0
            
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.replay_memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state
                self.total_steps += 1

                # Perform one step of the optimization (on the policy network)
                # Note difference in previous implementation -> cleared buffer after replay 
                # and waited until buffer size was reached instead of batch size
                # if len(self.replay_buffer) == self.replay_buffer_size:
                #     self.train_by_replay()
                #     self.replay_buffer.clear()
                self.train_by_replay()

                # Soft update of the target network's weights 
                # θ′ ← τ θ + (1 −τ )θ′
                # previous implementation updates were done for any episode where the reward is higher
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    #plot_durations(self)
                    break
                else:
                    self.check_model_improved += reward

            if self.check_model_improved > self.best_max:
                self.best_max = self.check_model_improved


    def train_by_replay(self):
        """
        TD update by replaying the history.
        """
        # step 1: generate replay samples (size = self.batch_size) from the replay buffer
        # e.g. uniform random replay or prioritize experience replay
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(self.BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        expectile_next = torch.zeros(self.BATCH_SIZE, self.action_dim, self.num_expectiles, device=self.device)
        with torch.no_grad():
            expectile_next[non_final_mask], _ = self.target_net(non_final_next_states)
            action_value_next = expectile_next[:, :, self.expectile_mean_idx]
            action_next = np.argmax(action_value_next, axis=1) 

            # choose the optimal expectile next
            expectile_next = expectile_next[np.arange(self.BATCH_SIZE), action_next, :]
        
            # The following part corresponds to Algorithm 2 in the paper
            # after getting the target expectile (or expectile_next), we need to impute the distribution
            # from the target expectile. This imputation step effectively re-generate the distribution
            # from the statistics (expectile)
            # Note that in the paper the authors assume dirac form to approximate a continuous PDF.
            # Therefore, the following steps generate several points on the x-axis of the PDF, each with an equal height
            # A visualization of this process is in figure 10 of the appendix section A of the paper
            z = self.imputation_strategy(expectile_next)

            # match the rewards and the discount rates from the memory to the same size as the expectile_next
            reshaped_reward_batch = np.tile(reward_batch.reshape(self.BATCH_SIZE, 1), (1, self.num_imputed_samples))
        
        discount_rate =  self.GAMMA * non_final_mask

        # TD update
        z = reshaped_reward_batch + discount_rate * z # NOTE check may need to still reshape discount rate if no broadcasting with final_mask
        
        _, optimal_action_expectiles = self.policy_net(state_batch)

        # Expectile regression loss
        loss = self.criterion(optimal_action_expectiles, z)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        """
        for each_ep in range(self.config.evaluate_episodes):
            state, info = self.env.reset() 
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            print('Episode: {} Reward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved[0].item(),
                                                                          self.best_max[0].item()))
            print('-' * 64)
            self.check_model_improved = 0

            for t in count():
                action = select_action(self, state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated 

                if render:
                    self.env.render()

                if done:
                    break
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)  
                    state = next_state
                    self.check_model_improved += reward

        
        print('Complete')
        # plot_durations(self, show_result=True)


    def select_action(self, state):

        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
            math.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # neural network returns expectile value
                # action value (Q): middle of all expectile values
                expectile_values, _ = self.policy_net.predict(state)
                action_value = expectile_values[0, :, self.expectile_mean_idx] 

                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return action_value.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)


    def imputation_strategy(self, expectile_next_batch):
        result_collection = np.zeros(shape=(self.batch_size, self.num_imputed_samples))
        start_vals = np.linspace(self.config.z_val_limits[0], self.config.z_val_limits[1], self.num_imputed_samples)
        
        for idx in range(self.batch_size):
            if self.imputation_method == "minimization":
                # To be discussed, I think this is pretty much problem-dependent
                # The bounds here limit the possible options of z
                # Having bounds could potentially avoid crazy z
                bnds = self.config.imputation_distribution_bounds
                optimization_results = minimize(self.minimize_objective_fc, args=(expectile_next_batch[idx, :]),
                                                x0=start_vals, bounds=bnds, method="SLSQP")
            elif self.imputation_method == "root":
                # the default root method is "hybr", it requires the input shape of x to be the same as
                # the output shape of the root results
                # in this case, it means that the imputed sample size is required to be exactly the same
                # as the number of expectiles
                optimization_results = root(self.root_objective_fc, args=(expectile_next_batch[idx, :]),
                                            x0=start_vals, method="hybr")
                result_collection[idx, :] = optimization_results.x
        return result_collection
        

    def minimize_objective_fc(self, x, expect_set):
        vals = 0
        for idx, each_expectile in enumerate(expect_set):
            diff = x - each_expectile
            diff = np.where(diff > 0, - self.cum_density[idx] * diff, (self.cum_density[idx] - 1) * diff)
            vals += np.square(np.mean(diff))

        return vals


    def root_objective_fc(self, x, expect_set):
        vals = []
        for idx, each_expectile in enumerate(expect_set):
            diff = x - each_expectile
            diff = np.where(diff > 0, - self.cum_density[idx] * diff, (self.cum_density[idx] - 1) * diff)
            vals.append(np.mean(diff))
        return vals
