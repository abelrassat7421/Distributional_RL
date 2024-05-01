# -*- coding:utf-8 -*-
from src.utils.config import Config
from src.utils.config_cartpole import ConfigCartpole
from src.utils.config_4statemdp import Config4statemdp

from src.agent.DQN import DQNAgent
from src.agent.CategoricalDQN import CategoricalDQNAgent


from src.environment import *

import gym

# We won't separate the agents from their neural network 
# architecture 


def get_config(env):
    env_name = env.spec.id
    config_classes = {
        "CartPole-v1": ConfigCartpole,
        "Simple4MDP-v0": Config4statemdp,
    }
    return config_classes.get(env_name, Config)(env)

def run_DQN_example(game):
    env = gym.make(game, render_mode="human")
    C = get_config(env)
    dqn_agent = DQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)

def run_CategoricalDQN_example(game):
    env = gym.make(game, render_mode="human")
    C = get_config(env)
    dqn_agent = CategoricalDQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)


if __name__ == '__main__':
    game = 'Simple4MDP-v0'
    # game = 'CartPole-v1'

    run_DQN_example(game)
    # run_CategoricalDQN_example(game)
    # run_QuantileDQN_example(game)
    # run_ExpectileDQN_example(game)
    # run_A2C_example(game)


