import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple
import geomloss
from matplotlib import pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adam, Optimizer
from Zernike.gymEnvironPtycho import ptychoEnv
# from pytorch3d.loss import chamfer_distance
from Zernike.znn import znn
from swd import swd
import itertools
from torchmetrics import StructuralSimilarityIndexMeasure

device = "cuda"
grid_size_in_A = 3
grid_size = grid_size_in_A*5

pixelOutput = False
numberOfAtoms = grid_size_in_A**2
if pixelOutput == True:
    label_dims = 0
    label_size = grid_size*grid_size
    shift = 0
    scaler = 1
else:
    label_dims = 2
    label_size = label_dims*numberOfAtoms
    shift = 0.5
    scaler = grid_size


class preCompute(nn.Module):
    def __init__(self, obs_size : int = 0, hidden_size: int = 2048):
        """Simple network that takes the Zernike moments and the last prediction as input and outputs a probability like map of atom positions.

        Args:
            obs_size: number of zernike moments
            n_actions: number of coordinates to predict (2 for x and y)
            hidden_size: size of hidden layers (should be big enough to store where atoms have been found, maybe 15x15)

        """
        super().__init__()

        self.gru = nn.GRU(input_size=obs_size, hidden_size=hidden_size, num_layers=5, batch_first=True)
        self.hidden_state = None

    def forward(self, zernikeValues_and_Pos, _ = None) -> Tensor:
        y , x = self.gru(zernikeValues_and_Pos) 
        x = x.transpose(0, 1).reshape((zernikeValues_and_Pos.shape[0], -1))


        # if self.hidden_state is None:
        #     y, hidden_state  = self.gru(zernikeValues_and_Pos)
        #     self.hidden_state = hidden_state.detach()
        # else:
        #     y, hidden_state  = self.gru(zernikeValues_and_Pos, self.hidden_state)
        #     self.hidden_state = hidden_state.detach()

        return  x

class preComputeTransformer(nn.Module):
    def __init__(self, obs_size : int = 0, hidden_size: int = 1024, numberOfHeads = 8):
        """Simple network that takes the Zernike moments and the last prediction as input and outputs a probability like map of atom positions.

        Args:
            obs_size: number of zernike moments
            hidden_size: size of hidden layers (should be big enough to store where atoms have been found, maybe 15x15)

        """
        super().__init__()
        
        # self.cls_token = Parameter(torch.randn(1, 1, obs_size, device = "cuda"))  # Learnable CLS token
        self.transformerEncode = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=obs_size, nhead=numberOfHeads, dim_feedforward=hidden_size, batch_first=True), num_layers=5)

    def forward(self, zernikeValues_and_Pos, mask = None) -> Tensor:
        # batch_size = zernikeValues_and_Pos.shape[0]

        # Expand CLS token to match batch size and prepend it
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, obs_size)
        # zernikeValues_and_Pos = torch.cat((cls_tokens, zernikeValues_and_Pos), dim=1)  # Add CLS token at the start

        y = self.transformerEncode(zernikeValues_and_Pos, src_key_padding_mask=mask)  # Apply transformer encoder
        y = y[:,0,:]

        return  y


class FinalLayer(nn.Module):
    def __init__(self, obs_size : int, output_size: int):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(obs_size,1000),
            #nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) 


class TwoPartLightning(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        numberOfPositions = 9,
        numberOfZernikeMoments = 40
    ) -> None:
        """Basic Transformer+Linear Model.

        Args:
            lr: learning rate
            numberOfPositions: number of positions in input

        """
        super().__init__()
    
        self.lr = lr
        
        self.obs_size = 0
        numberOfOSAANSIMoments = numberOfZernikeMoments
        for n in range(numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                self.obs_size += 1

        self.obs_size += 3 # 2 for x and y position of the agent
        self.nhead = 0
        for i in np.arange(8,20, 2):
            if self.obs_size % i == 0:
                self.nhead = i
                break
        if self.nhead == 0:
            for i in np.arange(2,8, 2):
                if self.obs_size % i == 0:
                    self.nhead = i
                    break
        print(f"number of heads: {self.nhead}")

        self.example_input_array = torch.zeros((1, numberOfPositions, self.obs_size), device=device, requires_grad=True)

        self.preComputeNN = preComputeTransformer(obs_size=self.obs_size, numberOfHeads=self.nhead)
        # self.preComputeNN = torch.compile(self.preComputeNN)
        self.finalLayerNN = FinalLayer(self.obs_size, label_size) 
        # self.finalLayerNN = torch.compile(self.finalLayerNN)

        if pixelOutput: self.loss_fct = nn.BCEWithLogitsLoss()#nn.MSELoss()
        else: self.loss_fct = geomloss.SamplesLoss()

        # self.finalLayer = znn(numberOfPositions*self.obs_size, label_size)#

    def forward(self, ptychoImages: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the coordinates of the atoms.

        Args:
            ptychoImages: current Zernike moments

        Returns:
            coordinates of the atoms ordered by x position

        """
        
        currentHiddenState = self.preComputeNN(ptychoImages ).flatten(start_dim=1)

        currentLabel :Tensor= self.finalLayerNN(currentHiddenState)
        # currentLabel : Tensor = self.finalLayer(ptychoImages.flatten(start_dim=1))

        labelOrdered = currentLabel
        if not pixelOutput:
            labelOrdered = currentLabel.flatten(start_dim=1).reshape((-1, numberOfAtoms, label_dims))
            sorted_indices = torch.argsort(labelOrdered[:, :, 0], dim=1)
            labelOrdered = torch.gather(labelOrdered, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, label_dims))*scaler

        return  (labelOrdered.flatten(start_dim=1) )

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.training_step(batch, log=False)
        self.log("val_loss", loss)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], log = True) -> Tensor:
        """Passes in a state x through the network and gets the coordinates of the atoms and computes the loss.

        Args:
            batch: current mini batch of replay data
            log: whether to log metrics

        Returns:
            Training loss

        """
        ptychoImages, atomPositionsLabel, mask = batch
        batch_size = ptychoImages.shape[0]

        currentHiddenState :Tensor= self.preComputeNN(ptychoImages, mask)[:,:].flatten(start_dim=1)

        currentLabel :Tensor= self.finalLayerNN(currentHiddenState)
        
        # currentLabel : Tensor = self.finalLayer(ptychoImages.flatten(start_dim=1))
        

        # loss = torch.nn.MSELoss()(currentLabel.flatten(start_dim=1), atomPositionsLabel.flatten(start_dim=1)/grid_size)
        if pixelOutput: loss = self.loss_fct(torch.clip(currentLabel.flatten(start_dim=1),0,1), atomPositionsLabel.flatten(start_dim=1))
        else: 
            currentLabelReshaped = currentLabel.flatten(start_dim=1).reshape((batch_size,-1,label_dims))
            atomPositionsLabelReshapedAndScaled = atomPositionsLabel.flatten(start_dim=1).reshape((batch_size,-1,label_dims))/scaler
            loss = torch.mean(self.loss_fct(currentLabelReshaped, atomPositionsLabelReshapedAndScaled))
            # lossGen = (sinkhorn(currentLabelReshaped[i], atomPositionsLabelReshapedAndScaled[i])[0] for i in range(len(currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims)))))
            # loss = torch.mean(torch.stack(list(lossGen)))

        # else: loss, _ = chamfer_distance(currentLabel.flatten(start_dim=1).reshape((-1,10,label_dims)), atomPositionsLabel.flatten(start_dim=1).reshape((-1,10,label_dims))/scaler)#, single_directional=True)
                                
        if log: self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.training_step(batch, log=False)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """Initialize Adam optimizer."""
        # params = []#self.dqn.parameters(), self.finalLayer.parameters()]
        # params = [self.finalLayer.parameters()]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
