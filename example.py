# -*- coding:utf-8 -*-
from src.utils.config import Config
from src.network.DQN_net import DQNNet
from src.agent.DQN import DQNAgent
import gym

# We won't separate the agents from their neural network 
# architecture 

def run_DQN_example(game):
    env = gym.make(game, render_mode="human")
    C = Config(env)
    dqn_agent = DQNAgent(config=C)
    dqn_agent.env = env

    dqn_agent.transition()
    print("finish training")
    print('=' * 64)
    print("evaluating.....")
    dqn_agent.eval_step(render=True)

if __name__ == '__main__':
    game = 'CartPole-v1'

    run_DQN_example(game)
    # run_CategoricalDQN_example(game)
    # run_QuantileDQN_example(game)
    # run_ExpectileDQN_example(game)
    # run_A2C_example(game)


