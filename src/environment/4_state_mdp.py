import gym
from gym import spaces
import numpy as np

from typing import Optional


class Simple4MDP(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode: Optional[str] = None):
        super(Simple4MDP, self).__init__()
        self.action_space = spaces.Discrete(1)  # Only one action available
        self.observation_space = spaces.Discrete(4)  # States 0, 1, 2, 3
        self.state = np.array([0])
        self.done = False

        # Define reward structure based on transitions
        self.rewards = {0: 2, 1: 2, 2: 1}
        self.alternate_rewards = {0: -2, 1: -2, 2: -1}

    def step(self, action):
        if self.done:
            return self.state, 0, self.done, {}

        # Decide the reward pattern randomly
        if np.random.rand() > 0.5:
            reward = self.rewards.get(self.state[0], 0)
        else:
            reward = self.alternate_rewards.get(self.state[0], 0)

        # Move to the next state
        self.state += 1

        # Check if terminal state
        if self.state[0] >= 3:
            self.done = True

        return self.state, reward, self.done, {}

    def reset(self):
        self.state = np.array([0])
        self.done = False
        return self.state, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass
