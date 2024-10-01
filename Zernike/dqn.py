import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
import torch
from IPython.core.display import display
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from Zernike.gymEnvironPtycho import ptychoEnv

device = "cuda"

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
            numberOfOSAANSIMoments = 10	
            for n in range(numberOfOSAANSIMoments + 1):
                for mShifted in range(2*n+1):
                    m = mShifted - n
                    if (n-m)%2 != 0:
                        continue
                    obs_size += 1
            obs_size += 2 # 2 for x and y position of the agent

        self.gru = nn.GRU(input_size=obs_size, hidden_size=hidden_size, num_layers=3, batch_first=True)

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=5,kernel_size=(3, 3)),   
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     # initialize second set of CONV => RELU => POOL layers
        #     nn.Conv2d(in_channels=5, out_channels=10,kernel_size=(5, 5)),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        # self.net = nn.Sequential(
        #     nn.Linear(obs_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 500),            
        # )

        self.finalLayer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 500),
            nn.ReLU(),
            nn.Linear(500, size_output)
        )

    def forward(self, x) -> Tensor:
        zernikeValues_and_Pos = x
        # currentWorldState = currentWorldState.unsqueeze(1)
        # cwsl = self.cnn(currentWorldState)
        #             #linear layers
        # cwsl = torch.flatten(cwsl, 1)
        # cwsl = nn.Linear(in_features=2560, out_features=500, device = device)(cwsl)
        y, _  = self.gru(zernikeValues_and_Pos)
        # y = torch.cat((cwsl, zv), 1)
        return self.finalLayer(y)


class Agent:
    def __init__(self, env:ptychoEnv) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        

    def reset(self, ptychoImages,atomProbabilityGrid) -> None:
        """Resents the environment and updates the state."""
        options = {"ptychoImages": ptychoImages, "atomProbabilityGrid": atomProbabilityGrid}
        self.zernikeObs_and_Pos, _ = self.env.reset(seed=0, options=options)
        # self.atomGridAlreadyVisited = np.zeros((75,75))
        batchSize = self.zernikeObs_and_Pos.shape[0]
        self.currentGrid  = torch.zeros((batchSize, 75, 75), device=device)
    

    def get_new_modeled_grid(self, net: nn.Module) -> "torch.Tensor":
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network

        Returns:
            action

        """

        output = net(self.zernikeObs_and_Pos)
        self.currentGrid = torch.clamp(output[:,:75*75].reshape(-1,75,75), min = 0, max = 1)

        if not torch.all(torch.isfinite(output)):
            raise Exception("Nan in output")
        newPosition = torch.round(torch.clamp(output[:,-2:] * 75, min = 0, max = 74)).to(dtype = torch.long)


        return newPosition
    
    def get_action(self, epsilon: float, newPosition : torch.Tensor) -> Tensor:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network

        Returns:
            action

        """
        
        if np.random.random() < epsilon:
            action = torch.randint(75, size=(newPosition.shape[0],2), device = device)
        else:
            action = newPosition

        # self.atomGridAlreadyVisited[action] = 999999

        return action

    #@torch.no_grad()
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

        newPosition = self.get_new_modeled_grid(net)
        action = self.get_action(epsilon, newPosition)

        # do step in the environment
        new_zernikeObs_and_Pos, reward, _ = self.env.step(action)

        self.zernikeObs_and_Pos = new_zernikeObs_and_Pos
        return reward, self.currentGrid

class DQNLightning(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        sync_rate: int = 10,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 1000,
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
        self.env = ptychoEnv()
        obs_shape = self.env.observation_space_size
        obs_size = np.prod(obs_shape)

        self.net = DQN()


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
        ptychoImages = x 
        atomProbabilityGrid = torch.zeros((x.shape[0], 75, 75), device=device)
        self.agent.reset(ptychoImages,atomProbabilityGrid)

        # step through environment with agent multple times
        for _ in range(self.episode_length):
            reward, modeledGrid = self.agent.play_step(self.net, epsilon= 0 )


        return modeledGrid.flatten(start_dim=1)

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

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> OrderedDict:
        loss = self.training_step(batch)
        self.log("val_loss", loss)
        return loss

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
            self.episode_reward += float(torch.mean(reward))
            self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = nn.MSELoss()(modeledGrid.flatten(), self.env.atomProbabilityGrid.flatten()/100)


                                   
        self.total_reward = self.episode_reward
        self.episode_reward = 0

        self.log_dict(
            {
                "reward": reward.mean(),
                "train_loss": loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return optimizer
