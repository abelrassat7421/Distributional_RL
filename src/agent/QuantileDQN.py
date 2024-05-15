# -*- coding:utf-8 -*-
from src.utils.replay_memory import ReplayMemory, Transition
from src.utils.visualization import plot_durations
from src.utils.reproducibility import set_seed
import numpy as np
import torch
import torch.optim as optim
from src.network.QuantileDQN_net import QuantileDQNNet, quantile_huber_loss
from itertools import count
import random
import math

class QuantileDQNAgent:

    def __init__(self, config):
        self.config = config 
        self.input_dim = config.input_dim
        self.action_dim = config.action_dim

        self.n_quantiles = config.num_quantiles
        self.quantile_weights = 1.0 / float(config.num_quantiles)
        self.k = config.huber_loss_threshold

        self.total_steps = 0
        self.num_episodes = config.num_episodes
        self.steps = config.steps
        self.BATCH_SIZE = config.BATCH_SIZE
        self.GAMMA = config.GAMMA
        self.LR = config.LR
        self.TAU = config.TAU
        self.device = config.device
        
        # reproducibility
        self.seed = config.seed 
        set_seed(self.seed)

        self.env = None
        # copying weights of base_net to policy_net and target_net
        self.policy_net = QuantileDQNNet(config)
        self.target_net = QuantileDQNNet(config)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        # If deemed helpful add base_lr and max_lr to hyperparameters 
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.001, mode="triangular2", cycle_momentum=False)
        self.criterion = quantile_huber_loss

        self.replay_buffer_size = config.replay_buffer_size
        self.replay_memory = ReplayMemory(self.replay_buffer_size)
        
        self.episode_durations = []
        self.check_model_improved = torch.tensor([0])
        self.best_max = torch.tensor([0])

        # for select action (epsilon-greedy)
        self.steps_done = 0
        
        # save for plotting evolution during training
        self.model_reward_hist = []
        self.model_loss_hist = []
       

    def transition(self):
        """
        In transition, the agent simply plays and records
        [current_state, action, reward, next_state, done]
        in the replay_memory

        Updating the weights of the neural network happens
        every single time the replay buffer size is reached.

        done: boolean, whether the game has ended or not.
        """
        self.env.action_space.seed(self.seed)

        for i_episode in range(self.num_episodes):
            state, info = self.env.reset(seed=self.seed) 
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            print('Episode: {} Reward: {} Max_Reward: {}'.format(i_episode, self.check_model_improved[0].item(), self.best_max[0].item()))
            print('-' * 64)
            self.check_model_improved = 0
            
            for t in count():
                action = self.select_action(state)
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
                # TODO see how to make differently than in Pytorch tutorial: optimize_model()
                # this is done partly in train_by_replay
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

                    self.model_reward_hist.append((i_episode, self.check_model_improved.detach().numpy()))
                    #plot_durations(self)
                    break
                else:
                    self.check_model_improved += reward

            if self.check_model_improved > self.best_max:
                self.best_max = self.check_model_improved
        
        cum_reward_per_episode = np.array([self.model_reward_hist[i][1] for i in range(len(self.model_reward_hist))])
        np.save('rewards_QDRL.npy', cum_reward_per_episode)
        np.save('losses_QDRL.npy', np.array(self.model_loss_hist))
        torch.save(self.policy_net.state_dict(), "policy_net_weights_QDRL.pth")
        torch.save(self.target_net.state_dict(), "target_net_weights_QDRL.pth")


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
        _, optimal_action_quantiles = self.policy_net(state_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        quantiles_next = torch.zeros(self.BATCH_SIZE, self.action_dim, self.n_quantiles, device=self.device)
        with torch.no_grad():
            quantiles_next[non_final_mask], _ = self.target_net(non_final_next_states)

        action_value_next = torch.sum(quantiles_next, dim=2)
        idx = torch.argmax(action_value_next, dim=1).to(torch.int64)
        #print("DEBUG #1:", idx.shape)

        quantiles_next = quantiles_next[torch.arange(self.BATCH_SIZE), idx, :]

        # match the rewards from the memory to the same size as prob_next
        reshaped_reward_batch = torch.tile(reward_batch.reshape(self.BATCH_SIZE, 1), (1, self.n_quantiles))

        # perform TD update 
        discount = torch.tile((self.GAMMA * non_final_mask).reshape(self.BATCH_SIZE, 1), (1, self.n_quantiles))

        # TD update
        quantiles_next = reshaped_reward_batch + discount * quantiles_next

        # Compute Huber loss 
        loss = self.criterion(optimal_action_quantiles, quantiles_next, self.k)
        self.model_loss_hist.append(loss.detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        # DEBUGGIING to find which tensor isn't tracked by the computation graph 
        # print("DEBUG #2:", loss.requires_grad, loss.grad_fn)
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.scheduler.step()
        print("Testing lr scheduler: self.scheduler.get_last_lr(): ", self.scheduler.get_last_lr())


    def eval_step(self, render=True):
        """
        Evaluation using the trained target network, no training involved
        :param render: whether to visualize the evaluation or not
        """
        for each_ep in range(self.config.evaluate_episodes):
            state, info = self.env.reset(seed=self.seed) 
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            print('Episode: {} Reward: {} Training_Max_Reward: {}'.format(each_ep, self.check_model_improved[0].item(),
                                                                          self.best_max[0].item()))
            print('-' * 64)
            self.check_model_improved = 0

            for t in count():
                action = self.select_action(state)
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


    # NOTE Create an abstract class of Agents with this method
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
            math.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # size = (1, action dimension, num_quantiles)
                action_quantiles, _ = self.policy_net(state)
                action_values = torch.sum(action_quantiles, dim=2)

                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                max_actions = action_values.max(1).indices.view(1, 1)
                return max_actions
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

