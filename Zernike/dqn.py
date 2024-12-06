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
from pytorch3d.loss import chamfer_distance
from Zernike.znn import znn
from swd import swd
from sinkhorn import sinkhorn
import itertools

device = "cuda"
grid_size = 15
pixelOutput = False
if pixelOutput == True:
    label_dims = 2
    label_size = grid_size*grid_size
    shift = 0
    scaler = 1
else:
    label_dims = 2
    label_size = label_dims*10
    shift = 0.5
    scaler = grid_size
#label_size  = 225

hidden_size = 2592


#scaler = torch.tensor([grid_size,grid_size,100], dtype = torch.float32, device = device)
#scaler = 100


class DQN(nn.Module):
    def __init__(self, obs_size : int = 0, hidden_size: int = hidden_size):
        """Simple network that takes the Zernike moments and the last prediction as input and outputs a probability like map of atom positions.

        Args:
            obs_size: number of zernike moments
            n_actions: number of coordinates to predict (2 for x and y)
            hidden_size: size of hidden layers (should be big enough to store where atoms have been found, maybe 75x75)

        """
        super().__init__()


        if obs_size == 0:
            numberOfOSAANSIMoments = 15
            for n in range(numberOfOSAANSIMoments + 1):
                for mShifted in range(2*n+1):
                    m = mShifted - n
                    if (n-m)%2 != 0:
                        continue
                    obs_size += 1
            #obs_size += 2 # 2 for x and y position of the agent

        self.gru = nn.GRU(input_size=obs_size, hidden_size=hidden_size, num_layers=3, batch_first=True)
        self.hidden_state = None

    def forward(self, x) -> Tensor:
        zernikeValues_and_Pos = x
        y, _ = self.gru(zernikeValues_and_Pos)
        # if self.hidden_state is None:
        #     y, hidden_state  = self.gru(zernikeValues_and_Pos)
        #     self.hidden_state = hidden_state.detach()
        # else:
        #     y, hidden_state  = self.gru(zernikeValues_and_Pos, self.hidden_state)
        #     self.hidden_state = hidden_state.detach()

        return  y

# class DQN(nn.Module):
#     def __init__(self, obs_size : int = 0, hidden_size: int = hidden_size):
#         """Simple network that takes the Zernike moments and the last prediction as input and outputs a probability like map of atom positions.

#         Args:
#             obs_size: number of zernike moments
#             n_actions: number of coordinates to predict (2 for x and y)
#             hidden_size: size of hidden layers (should be big enough to store where atoms have been found, maybe 75x75)

#         """
#         super().__init__()


#         if obs_size == 0:
#             numberOfOSAANSIMoments = 15
#             for n in range(numberOfOSAANSIMoments + 1):
#                 for mShifted in range(2*n+1):
#                     m = mShifted - n
#                     if (n-m)%2 != 0:
#                         continue
#                     obs_size += 1
#             #obs_size += 2 # 2 for x and y position of the agent
        
#         self.conv1 = nn.Conv2d(in_channels=136, out_channels=64, kernel_size=3, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(32)
#         # self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)  # Conv1 + ReLU + BatchNorm

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)  # Conv2 + ReLU + BatchNorm

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = torch.relu(x)  # Conv3 + ReLU + BatchNorm

#         # x = self.pool(x)  # Global Average Pooling
#         x = torch.flatten(x, 1)  # Flatten for fully connected layer
#         return x


class FinalLayer(nn.Module):
    def __init__(self, output_size: int):
        super().__init__() 
        self.net = nn.Sequential(
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) 

class Agent:
    def __init__(self, env:ptychoEnv) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        

    def reset(self, ptychoImages,atomPositionsLabel) -> None:
        """Resents the environment and updates the state."""
        options = {"ptychoImages": ptychoImages, "atomPositionsLabel": atomPositionsLabel}
        self.zernikeObs_and_Pos, _ = self.env.reset(seed=0, options=options)

        batchSize = self.zernikeObs_and_Pos.shape[0]

        self.currentLabelPred = torch.zeros((batchSize, hidden_size), device=device)
    

    def get_new_label_pred(self, net: nn.Module) -> None:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network

        Returns:
            action

        """

        output = net(torch.cat((self.zernikeObs_and_Pos,self.currentLabelPred),dim=1))
        self.currentLabelPred = output

        # if not torch.all(torch.isfinite(output)):
        #     raise Exception("Nan in output")
    
    
    def get_action(self, epsilon: float, newPosition : torch.Tensor, stepNr : int) -> Tensor:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network

        Returns:
            action

        """

        # actions = [(x,y) for x in np.arange(grid_size)[::5] for y in np.arange(grid_size)[::5]]

        #create a torch tensor of shape (batchSize, 2) with action from action at stepNr
        # action = torch.tensor([actions[stepNr % len(actions)] for _ in range(newPosition.shape[0])], device = device))
        # if np.random.random() < epsilon:
        #     action = torch.randint(grid_size, size=(newPosition.shape[0],2), device = device)
        # else:
        #     action = newPosition
        action = newPosition

        # self.atomGridAlreadyVisited[action] = 999999

        return action

    
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        stepNr: int = 0
    ):
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done

        """

        self.get_new_label_pred(net)
        # action = self.get_action(1.1, self.currentLabelPred, stepNr)

        # do step in the environment
        # new_zernikeObs_and_Pos, reward, _ = self.env.step(action)

        new_zernikeObs_and_Pos, reward, _ = self.env.step_simple(stepNr)

        self.zernikeObs_and_Pos = new_zernikeObs_and_Pos
        return reward, self.currentLabelPred

class DQNLightning(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        sync_rate: int = 10,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 9,
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

        self.agent = Agent(self.env)
        self.total_reward = 0
        self.episode_reward = 0

        self.obs_size = 0
        numberOfOSAANSIMoments = 15
        for n in range(numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                self.obs_size += 1
        self.example_input_array = torch.zeros((1, 9, self.obs_size), device=device)

        self.dqn = DQN()
        #self.finalLayer = znn(9*self.obs_size, label_size)#
        self.finalLayer = FinalLayer(label_size) 

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        ptychoImages = x
        atomPositionsLabel = torch.zeros((x.shape[0], label_size), device=device)
        # self.agent.reset(ptychoImages,atomPositionsLabel)

        # step through environment with agent multiple times
        # for stepNr in range(self.episode_length):
        #     reward, currentLabelPred = self.agent.play_step(self.net, epsilon= 0, stepNr=stepNr )
        currentHiddenState = self.dqn(x )[:,:].flatten(start_dim=1)
        currentLabel :Tensor= self.finalLayer(currentHiddenState)
        # currentLabel : Tensor = self.finalLayer(x.flatten(start_dim=1))

        labelOrdered = currentLabel
        if not pixelOutput:
            labelOrdered = currentLabel.flatten(start_dim=1).reshape((-1, 10, label_dims)) + shift
            sorted_indices = torch.argsort(labelOrdered[:, :, 0], dim=1)
            labelOrdered = torch.gather(labelOrdered, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, label_dims))

        return  (labelOrdered.flatten(start_dim=1) )*scaler

    def get_epsilon(self, start, end, frames) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.training_step(batch, log=False)
        self.log("val_loss", loss)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], log = True) -> Tensor:
        """Carries out an episode worth of steps through the environment using the DQN

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        x, y = batch
        ptychoImages = x 
        atomPositionsLabel = y
        
        # print(atomPositionsLabel)
        # exit()
        # self.agent.reset(ptychoImages,atomPositionsLabel)
        epsilon = 0#self.get_epsilon(self.eps_start, self.eps_end, self.eps_last_frame)
        self.log("epsilon", epsilon)

        # step through environment with agent multiple times
        # for stepNr in range(self.episode_length):
        #     reward, currentHiddenState = self.agent.play_step(self.net, epsilon, stepNr)
        #     self.episode_reward += 0
        currentHiddenState :Tensor= self.dqn(x)[:,:].flatten(start_dim=1)
        currentLabel :Tensor= self.finalLayer(currentHiddenState)
        # currentLabel : Tensor = self.finalLayer(x.flatten(start_dim=1))

        # currentLabelReshaped = currentLabel.reshape((-1,int(label_size/2),2))
        # indices = torch.argsort(currentLabelReshaped[:,:,0])


        # loss = torch.nn.MSELoss()(currentLabel.flatten(start_dim=1), atomPositionsLabel.flatten(start_dim=1)/grid_size-0.5)
        if pixelOutput: loss = swd(currentLabel.flatten(start_dim=1).reshape((-1,1,grid_size,grid_size)), atomPositionsLabel.flatten(start_dim=1).reshape((-1,1,grid_size,grid_size))/scaler,  device=device)
        # else: 
        #     lossGen = (sinkhorn(currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims))[i], atomPositionsLabel.flatten(start_dim=1).reshape((-1,10,label_dims))[i]/scaler-0.5)[0] for i in range(10))
        #     loss = torch.mean(torch.stack(list(lossGen)))

        else: loss, _ = chamfer_distance(currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims)), atomPositionsLabel.flatten(start_dim=1).reshape((-1,10,label_dims))/scaler-0.5, single_directional=True)
        
                                   
        self.total_reward = 0
        self.episode_reward = 0
        if log: self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.training_step(batch, log=False)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        params = [self.finalLayer.parameters()]#self.dqn.parameters(), self.finalLayer.parameters()]
        optimizer = torch.optim.AdamW(itertools.chain(*params), lr=self.lr)
        return optimizer
