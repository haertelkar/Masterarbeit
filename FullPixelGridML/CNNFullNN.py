import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import geomloss
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
import torch
from IPython.core.display import display
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from FullPixelGridML.cnn import cnn
from Zernike.gymEnvironPtycho import ptychoEnv
# from pytorch3d.loss import chamfer_distance
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
#label_size  = 225t
channels = 3*3

class TwoPartLightningCNN(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
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
        self.cnn = cnn(channels,20)
        self.example_input_array = torch.zeros((1, channels, 20, 20), device=device)
        self.loss_fct = geomloss.SamplesLoss()


    def forward(self, ptychoImages: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the coordinates of the atoms.

        Args:
            ptychoImages: current Zernike moments

        Returns:
            coordinates of the atoms ordered by x position

        """

        currentHiddenState = self.cnn(ptychoImages)

        labelOrdered = currentHiddenState
        if not pixelOutput:
            labelOrdered = currentHiddenState.flatten(start_dim=1).reshape((-1, 10, label_dims))
            sorted_indices = torch.argsort(labelOrdered[:, :, 0], dim=1)
            labelOrdered = torch.gather(labelOrdered, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, label_dims))

        return  (labelOrdered.flatten(start_dim=1) )*scaler

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.training_step(batch, log=False)
        self.log("val_loss", loss)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], log = True) -> Tensor:
        """Passes in a state x through the network and gets the coordinates of the atoms and computes the loss.

        Args:
            batch: current mini batch of replay data
            log: whether to log metrics

        Returns:
            Training loss

        """
        ptychoImages, atomPositionsLabel = batch

        currentLabel = self.cnn(ptychoImages)


        # loss = torch.nn.MSELoss()(currentLabel.flatten(start_dim=1), atomPositionsLabel.flatten(start_dim=1)/grid_size)
        if pixelOutput: loss = swd(currentLabel.flatten(start_dim=1).reshape((-1,1,grid_size,grid_size)), atomPositionsLabel.flatten(start_dim=1).reshape((-1,1,grid_size,grid_size))/scaler,  device=device)
        else: 
            currentLabelReshaped = currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims))
            atomPositionsLabelReshapedAndScaled = atomPositionsLabel.flatten(start_dim=1).reshape((-1,10,label_dims))/scaler
            loss = torch.mean(self.loss_fct(currentLabelReshaped, atomPositionsLabelReshapedAndScaled))

            # lossGen = (sinkhorn(currentLabelReshaped[i], atomPositionsLabelReshapedAndScaled[i])[0] for i in range(len(currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims)))))
            # loss = torch.mean(torch.stack(list(lossGen)))

        # else: loss, _ = chamfer_distance(currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims)), atomPositionsLabel.flatten(start_dim=1).reshape((-1,10,label_dims))/scaler)#, single_directional=True)
                                
        if log: self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.training_step(batch, log=False)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.AdamW(self.cnn.parameters(), lr=self.lr)
        return optimizer
