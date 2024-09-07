import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import glob
import h5py

import pandas as pd

class pytchoEnv(gym.Env):
    def __init__(self):
        super(pytchoEnv, self).__init__()
        self.atomProbabilityGrid = np.zeros((75,75))  # Grid of possible scan positions represented as a 2D numpy array with zero everywhere without atoms
        self.num_rows, self.num_cols = self.atomProbabilityGrid.shape
        self.start_pos  = (int(self.num_rows/2),int(self.num_cols/2))
        self.current_pos = self.start_pos #starting position is current position of agent

        # 75x75 grid
        self.action_space = spaces.MultiDiscrete([self.num_rows, self.num_cols])  
        
        numberOfZernikeMoments = 0
        numberOfOSAANSIMoments = 20	
        for n in range(numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                numberOfZernikeMoments += 1
        # Observation space is current coordinates plus Zernike vector
        self.observation_space = spaces.Discrete(numberOfZernikeMoments)
        self.allZernikeMoments = np.zeros((self.num_rows, self.num_cols, numberOfZernikeMoments))

    def reset(self, seed, options):
        super().reset(seed=seed)
        ptychoImages, atomProbabilityGrid = options["ptychoImages"], options["atomProbabilityGrid"]
        self.current_pos = self.start_pos
        self.atomProbabilityGrid = atomProbabilityGrid
        self.allZernikeMoments = ptychoImages
        return self.current_pos

    def step(self, action):
        new_pos = action


        self.current_pos = new_pos

        # Reward function
        reward = self.atomProbabilityGrid[self.current_pos[0], self.current_pos[1]]
		
        

        return self.allZernikeMoments[self.current_pos], reward, {}

    # def _is_valid_position(self, pos):
    #     row, col = pos
   
    #     # If agent goes out of the grid
    #     if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
    #         return False

    #     # If the agent hits an obstacle
    #     if self.scanGrid[row, col] == '#':
    #         return False
    #     return True