import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from IPython.core.display import display
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from gymEnvironPtycho import pytchoEnv

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class DQN(nn.Module):
    def __init__(self, obs_size : int = 0, size_output: int = 75*75 + 2, hidden_size: int = 250):
        """Simple network that takes the Zernike moments and the last prediction as input and outputs a probability like map of atom positions.

        Args:
            obs_size: number of zernike moments
            n_actions: number of coordinates to predict (2 for x and y)
            hidden_size: size of hidden layers (should be big enough to store where atoms have been found, maybe 75x75)

        """
        super().__init__()


        if obs_size == 0:
            numberOfOSAANSIMoments = 20	
            for n in range(numberOfOSAANSIMoments + 1):
                for mShifted in range(2*n+1):
                    m = mShifted - n
                    if (n-m)%2 != 0:
                        continue
                    obs_size += 1

        

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5,kernel_size=(3, 3)),   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # initialize second set of CONV => RELU => POOL layers
            nn.Conv2d(in_channels=5, out_channels=10,kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #linear layers
            nn.Linear(in_features=75*75*10, out_features=500))

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 500),
        )

        self.finalLayer = nn.Linear(in_features=1000, out_features=size_output)
        
    def forward(self, x):
        currentWorldState, zernikeValues = x
        cwsl = self.cnn(currentWorldState)
        zv = self.net(zernikeValues)
        x = torch.cat((cwsl, zv), 1)
        return torch.reshape(self.finalLayer(x),(75,75))


class Agent:
    def __init__(self, env:gym.Env) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.state = self.env.reset()
        

    def reset(self, ptychoImages,atomProbabilityGrid) -> None:
        """Resents the environment and updates the state."""
        options = {"ptychoImages": ptychoImages, "atomProbabilityGrid": atomProbabilityGrid}
        self.state = self.env.reset(seed=0, options=options)
        self.atomGridAlreadyVisited = np.zeros_like(atomProbabilityGrid)
        self.currentGrid  = np.zeros((75,75))
    

    def get_new_modeled_grid(self, net: nn.Module):
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network

        Returns:
            action

        """

        output = net([self.state, self.currentGrid])
        newGrid = output[:75*75]
        newPosition = (output[-2:] * 75).to(int)


        return newGrid, newPosition
    
    def get_action(self, epsilon: float, newPosition : torch.Tensor):
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network

        Returns:
            action

        """

        if np.random.random() < epsilon:
            action = np.random.randint(75, size=2)
        else:
            action = newPosition.numpy()
        
        self.atomGridAlreadyVisited[action] = 999999

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
    ):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done

        """

        modeledGrid, newPosition = self.get_new_modeled_grid(net)
        action = self.get_action(epsilon, newPosition)

        # do step in the environment
        new_state, reward, terminated, truncated, _ = self.env.step(action)

        self.state = new_state
        return float(reward), modeledGrid

class DQNLightning(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        sync_rate: int = 10,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
    ) -> None:
        """Basic DQN Model.

        Args:
            lr: learning rate
            sync_rate: how many frames do we update the target network
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode

        """
        super().__init__()
    
        self.lr = lr
        self.sync_rate = sync_rate
        self.eps_last_frame =  eps_last_frame
        self.eps_start = eps_start
        self.eps_end=eps_end
        self.episode_length=episode_length
        self.env = pytchoEnv()
        obs_shape = self.env.observation_space.shape[0] 
        obs_size = np.prod(obs_shape)
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)


        self.agent = Agent(self.env)
        self.total_reward = 0
        self.episode_reward = 0

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.net(x)
        return output

    def mse_loss(self) -> Tensor:
        """Calculates the mse loss as a difference between the maximum reward possible in one step and the current reward

        Returns:
            loss

        """

        return nn.MSELoss()(self.env.atomProbabilityGrid.flatten(), self.net(torch.tensor(self.env.atomProbabilityGrid).float().flatten()))

    def get_epsilon(self, start, end, frames) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor]) -> OrderedDict:
        """Carries out an episode worth of steps through the environment using the DQN

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        x, y = batch
        ptychoImages = x 
        atomProbabilityGrid = y
        self.agent.reset(ptychoImages,atomProbabilityGrid)
        epsilon = self.get_epsilon(self.eps_start, self.eps_end, self.eps_last_frame)
        self.log("epsilon", epsilon)

        # step through environment with agent multple times
        for _ in range(self.episode_length):
            reward, modeledGrid = self.agent.play_step(self.net, epsilon)
            self.episode_reward += reward
            self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = nn.MSELoss()(modeledGrid.flatten(), torch.tensor(self.env.atomProbabilityGrid).float().flatten())


                                   
        self.total_reward = self.episode_reward
        self.episode_reward = 0

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return optimizer
