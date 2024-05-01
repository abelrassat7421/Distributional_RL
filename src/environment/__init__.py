# -*- coding:utf-8 -*-
from .five_state_MDP import *
from gym.envs.registration import register

register(
    id='fiveStateMDP-v0',
    max_episode_steps=200,
    entry_point='src.environment.five_state_MDP:fiveStateMDP',
    reward_threshold=10000.0
)

register(
    id='Simple4MDP-v0',
    max_episode_steps=200,
    entry_point='src.environment.4_state_mdp:Simple4MDP',
    reward_threshold=1000.0
)
