# -*- coding:utf-8 -*-
from src.utils.config import Config
from src.utils.reproducibility import set_seed
from src.agent.DQN import DQNAgent
from src.agent.CategoricalDQN import CategoricalDQNAgent
from src.agent.ExpectileDQN import ExpectileDQNAgent

import gym

# We won't separate the agents from their neural network 
# architecture 

def run_DQN_example(game):
    set_seed(42)
    env = gym.make(game, render_mode="human")
    C = Config(env)
    dqn_agent = DQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)

def run_CategoricalDQN_example(game):
    set_seed(42)
    env = gym.make(game, render_mode="human")
    C = Config(env)
    dqn_agent = CategoricalDQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)

def run_ExpectileDQN_example(game):
    set_seed(42)
    env = gym.make(game, render_mode="human")
    C = Config(env)
    dqn_agent = ExpectileDQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)

if __name__ == '__main__':
    game = 'CartPole-v1'

    # run_DQN_example(game)
    run_CategoricalDQN_example(game)
    # run_QuantileDQN_example(game)
    # run_ExpectileDQN_example(game)


