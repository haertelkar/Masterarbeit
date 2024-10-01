import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import glob
import h5py

import pandas as pd
import torch

device = "cuda"

class ptychoEnv(gym.Env):
    def __init__(self):
        super(ptychoEnv, self).__init__()
        self.atomProbabilityGrid = torch.tensor((75,75), device = device).unsqueeze(0)
        self.num_rows, self.num_cols = 75,75
        self.start_pos  = torch.tensor((int(self.num_rows/2),int(self.num_cols/2)), dtype= torch.long, device = device).unsqueeze(0) 
        self.current_pos = None

        # 75x75 grid
        self.action_space = spaces.MultiDiscrete([self.num_rows, self.num_cols])  
        
        numberOfZernikeMoments = 0
        numberOfOSAANSIMoments = 10	
        for n in range(numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                numberOfZernikeMoments += 1
        # Observation space is current coordinates plus Zernike vector
        self.observation_space_size = numberOfZernikeMoments 
        self.observation_space = spaces.Discrete(self.observation_space_size)
        self.allZernikeMoments = torch.zeros((self.num_rows, self.num_cols, numberOfZernikeMoments), device=device).unsqueeze(0)

    def reset(self, seed, options:"dict[str, torch.Tensor]") -> "tuple[torch.Tensor, dict]":
        super().reset(seed=seed)
        ptychoImages, atomProbabilityGrid = options["ptychoImages"], options["atomProbabilityGrid"]
        self.current_pos = self.start_pos.repeat((ptychoImages.shape[0],1))
        self.atomProbabilityGrid = atomProbabilityGrid.reshape((-1,75,75))
        self.allZernikeMoments = ptychoImages.reshape((ptychoImages.shape[0],75,75,-1)).to(device)

        return torch.cat([self.allZernikeMoments[torch.arange(self.allZernikeMoments.shape[0]),self.current_pos[:,0], self.current_pos[:,1]], self.current_pos], dim = -1), {}

    def step(self, action: torch.Tensor):
        self.current_pos = action

        # Reward function
        batch_size = self.allZernikeMoments.shape[0]

        reward = self.atomProbabilityGrid[torch.arange(batch_size),self.current_pos[:,0], self.current_pos[:,1]]

        return torch.cat([self.allZernikeMoments[torch.arange(batch_size),self.current_pos[:,0], self.current_pos[:,1]], self.current_pos], dim = -1), reward, {}

    # def _is_valid_position(self, pos):
    #     row, col = pos
   
    #     # If agent goes out of the grid
    #     if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
    #         return False

    #     # If the agent hits an obstacle
    #     if self.scanGrid[row, col] == '#':
    #         return False
    #     return True