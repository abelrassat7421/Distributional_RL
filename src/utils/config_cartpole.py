# -*- coding:utf-8 -*-
import torch 

from .config import Config


# TODO add a parameter for class initialisation for the type of game 
# and automatically extract environment parameters -> for now setup 
# for CartPole environment
class ConfigCartpole(Config):
    def __init__(self, env):
        # environment parameters
        self.action_dim = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        self.input_dim = len(state)

        self.num_episodes = 300
        self.evaluate_episodes = 50
        # note that OpenAI gym has max environment steps (e.g. max_step = 200 for CartPole)
        self.steps = 500  # Note this may have changed to 500 TODO (check)
        self.replay_buffer_size = 10000

        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 100 #1000
        self.TAU = 0.005
        self.LR = 1e-3 # 1e-4

        # Neural network parameters
        # define optimizer in the Neural net class? TODO optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        # NOTE: In the original DRL repo implemented with tensorflow there are:
        # 1) checkpoints which basically periodically saves the model weights during training 
        # e.g., whenever the model improves 
        #     def save_model_checkpoint(epoch, model, filepath='harvey.pth'):
        #          torch.save(model.state_dict(), filepath)
        # 2) Early stopping: this can be implemented with a EarlyStopping class for example

        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------------------------------------------------
        #                Specilaized Parameters for each RL Algorithm 
        # -----------------------------------------------------------------------


        # Categorical DQN parameters
        self.categorical_Vmin = 0 # TODO probably link to self.z_val_limits for expectile code
        self.categorical_Vmax = 70
        self.categorical_n_atoms = 10  

        # Quantile Regression DQN parameters
        self.num_quantiles = 20
        self.huber_loss_threshold = 1.0
        
        # Expetile ER-DQN parameters
        self.num_expectiles = 10 
        self.z_val_limits = (0, 10) # NOTE change depending on the environment?
        self.num_imputed_samples = 10
        self.imputation_distribution_bounds = tuple(self.z_val_limits for _ in range(self.num_imputed_samples))
        self.imputation_method = "root"  # root or minimization

        # Imputation used to reconstruct a distribution from its distribution statistics e.g., quatiles 
        # Introduced in the general DRL framework "Statistics and Samples in Distributional Reinforcement Learning"
        # the default root method is "hybr", it requires the input shape of x to be the same as
        # the output shape of the root results
        # in this case, it means that the imputed sample size to be exactly the same
        # as the number of expectiles
        # this is also the assumption in the paper if you look closely at Algorithm 2 and appendix D.1
        if self.imputation_method == "root":
            assert self.num_expectiles == self.num_imputed_samples, \
                "if you use root method, the number of imputed samples must be equal to the number of expectiles"

        # a2c parameters
        self.head_out_dim = 20
        
        # Laplace Code parameters
        self.num_gamma = 50
        self.num_sensitivities = 10
        self.learning_rate = 0.1  # not using neural networks for now
        self.alpha = 1 # Regularization parameter for SVD-based Discrete Linear Decoder




        
