from typing import Tuple
import geomloss
from matplotlib import pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
# from pytorch3d.loss import chamfer_distance

device = "cuda"
grid_size_in_A = 3 #is window size in Angstroms
grid_size = grid_size_in_A*5

pixelOutput = False

def create_quarter_circle_masks(patch_size: int, image_size: int, device):
    """
    Creates boolean masks for 4 quarter-circle patches based on the full image center.
    Each patch is a 19x19 region of a 38x38 image, but only pixels within the circle centered
    at (19,19) of radius 19 are kept. This is useful, because only the inner pixels are interesting
    """
    center = image_size // 2 - 0.5  # 18.5 (patch geht von 0 bis 18, also 19x19)
    r = center
    
    grid_x, grid_y = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
    masks = []
    mask_total = torch.zeros((image_size, image_size), dtype=torch.bool, device=device)
    mask_total = (grid_x - center) ** 2 + (grid_y - center) ** 2 <= r ** 2
    for x in range(2):
        for y in range(2):
            mask = torch.zeros((patch_size, patch_size), dtype=torch.bool, device=device)
            mask = mask_total[x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size].flatten().clone()
            masks.append(mask)

    return masks  # List of 4 boolean masks [19, 19]

class PatchEncoderWithMeta(nn.Module):
    def __init__(self, patch_size=19, emb_dim=64, image_size=38):   
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.image_size = image_size
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (self.image_size//self.patch_size)**2  # Since 38x38 is split into 4x 19x19
        assert self.num_patches == 4, "Expected 4 patches, more are not supported in this implementation"
        self.quarter_circle_masks = create_quarter_circle_masks(patch_size, image_size, device)
        self.patch_area  = self.quarter_circle_masks[0].sum().item()  # Number of pixels in the quarter circle patch
        assert self.quarter_circle_masks[0].sum() == self.quarter_circle_masks[1].sum() and self.quarter_circle_masks[1].sum() == self.quarter_circle_masks[2].sum() and self.quarter_circle_masks[2].sum() == self.quarter_circle_masks[3].sum(), \
            f"Expected all equal patch areas, but got {self.quarter_circle_masks[0].sum()}, {self.quarter_circle_masks[1].sum()}, {self.quarter_circle_masks[2].sum()}, {self.quarter_circle_masks[3].sum()}"
        self.linear_proj = nn.Linear(self.patch_area, emb_dim)  # [361] -> [D]


    def forward(self, images:torch.Tensor, metadata:torch.Tensor) -> torch.Tensor:
        """
        images:   [B, 1444]  (flattened 38x38)
        metadata: [B, T, 2]
        """
        num_images, _ = images.shape

        # Step 1: Reshape to [B*T, 1, 38, 38]
        x = images.view(num_images, 1, self.image_size, self.image_size)

        # Step 2: Split into 4 non-overlapping 19x19 patches
        patches = torch.cat([
            x[:, :, 0:self.patch_size, 0:self.patch_size].flatten(start_dim=2)[:,:,self.quarter_circle_masks[0]],     # Top-left
            x[:, :, 0:self.patch_size, self.patch_size:self.image_size].flatten(start_dim=2)[:,:,self.quarter_circle_masks[1]],    # Top-right
            x[:, :, self.patch_size:self.image_size, 0:self.patch_size].flatten(start_dim=2)[:,:,self.quarter_circle_masks[2]],    # Bottom-left
            x[:, :, self.patch_size:self.image_size, self.patch_size:self.image_size].flatten(start_dim=2)[:,:,self.quarter_circle_masks[3]],   # Bottom-right
        ], dim=1)  # Now [B*T, 4, self.patch_area]

        # Step 4: Linear projection → [B*T, 4, emb_dim]
        patch_embeddings = self.linear_proj(patches)

        # Step 5: Prepare metadata and patch indices
        metadata = metadata.unsqueeze(-2).expand(-1, self.num_patches, -1)  # [num_images, 4, 2]
        metadata = metadata.contiguous().view(num_images, self.num_patches, 2)      # [B*T, 4, 2]

        patch_indices = torch.arange(0, self.num_patches, device=images.device).float()  # [4]
        patch_indices = patch_indices.view(1, self.num_patches, 1).expand(num_images, -1, -1) # [B*T, 4, 1]

        # Step 6: Concatenate [embedding | metadata | patch index]
        out = torch.cat([patch_embeddings, metadata, patch_indices], dim=-1)  # [B*T, 4, D+3]

        # Step 7: Reshape back to [numImages*4, D+3]
        out = out.view(-1, self.emb_dim + 3)
        return out

class preCompute(nn.Module):
    def __init__(self, obs_size : int = 0, hidden_size: int = 1024):
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
    def __init__(self, obs_size : int = 0, hidden_size: int = 1024, numberOfHeads = 8, num_layers = 5, cls_like = True):
        """Simple network that takes the Zernike moments and the last prediction as input and outputs a probability like map of atom positions.

        Args:
            obs_size: number of zernike moments
            hidden_size: size of hidden layers 

        """
        super().__init__()
        self.cls_like = cls_like
        # self.cls_token = Parameter(torch.randn(1, 1, obs_size, device = "cuda"))  # Learnable CLS token
        self.transformerEncode = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=obs_size, nhead=numberOfHeads, dim_feedforward=hidden_size, batch_first=True), num_layers=num_layers)

    def forward(self, zernikeValues_and_Pos, mask = None) -> Tensor:
        # batch_size = zernikeValues_and_Pos.shape[0]

        # Expand CLS token to match batch size and prepend it
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, obs_size)
        # zernikeValues_and_Pos = torch.cat((cls_tokens, zernikeValues_and_Pos), dim=1)  # Add CLS token at the start

        x = self.transformerEncode(zernikeValues_and_Pos, src_key_padding_mask=mask)  # Apply transformer encoder
        if self.cls_like:
            # If cls_like, return the first token representation
            return x[:, 0, :]
        
        if mask is not None:
            # Invert mask for calculation: False for padded positions
            # Unsqueeze to allow broadcasting: [B, T*4] -> [B, T*4, 1]
            inverted_mask = ~mask.unsqueeze(-1)

            # Zero out the padded values to not include them in the sum
            x_masked = x * inverted_mask

            # Sum the features of valid tokens
            sum_x = x_masked.sum(dim=1)

            # Count the number of valid tokens for each item in the batch
            lengths = inverted_mask.sum(dim=1)

            # Avoid division by zero for empty sequences
            lengths = lengths.clamp(min=1)

            # Return the mean of the valid tokens
            return sum_x / lengths
        else:
            # If no mask, just take the mean over the sequence dimension
            return x.mean(dim=1)
        #pooling somehow performs worse than just using the first hidden state
class FinalLayer(nn.Module):
    def __init__(self, obs_size : int, output_size: int, num_layers: int = 3):
        super().__init__() 
        inbetween_layers = [
            nn.Linear(obs_size,1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU()
            ]
        
        for _ in range(num_layers - 3):
            inbetween_layers.append(nn.Linear(500, 500))
            inbetween_layers.append(nn.ReLU())

        inbetween_layers.append(nn.Linear(500, output_size))
        self.net = nn.Sequential(
            *inbetween_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) 


class TwoPartLightning(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        numberOfPositions = 9,
        numberOfZernikeMoments = 40,
        numberOfAtoms = grid_size_in_A**2,
        hidden_size: int = 1024, 
        num_layers = 5,
        fc_num_layers = 3
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

        self.obs_size += 3 # 2 for x and y position 
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
        self.numberOfAtoms = numberOfAtoms
        if pixelOutput == True:
            self.label_dims = 0
            label_size = grid_size*grid_size
            self.scaler = 1
        else:
            self.label_dims = 2
            label_size = self.label_dims*numberOfAtoms
            self.scaler = grid_size



        self.preComputeNN = preComputeTransformer(obs_size=self.obs_size, numberOfHeads=self.nhead, hidden_size=hidden_size, num_layers=num_layers, cls_like=True)
        # self.preComputeNN = torch.compile(self.preComputeNN)
        self.finalLayerNN = FinalLayer(self.obs_size, label_size, num_layers = fc_num_layers) 
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
            labelOrdered = currentLabel.flatten(start_dim=1).reshape((-1, self.numberOfAtoms, self.label_dims))
            sorted_indices = torch.argsort(labelOrdered[:, :, 0], dim=1)
            labelOrdered = torch.gather(labelOrdered, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, self.label_dims))*self.scaler

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
            currentLabelReshaped = currentLabel.flatten(start_dim=1).reshape((batch_size,-1,self.label_dims))
            atomPositionsLabelReshapedAndScaled = atomPositionsLabel.flatten(start_dim=1).reshape((batch_size,-1,self.label_dims))/self.scaler
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
    

class ThreePartLightning(LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        numberOfAtoms = grid_size_in_A**2
    ) -> None:
        """Basic Transformer+Linear Model.

        Args:
            lr: learning rate

        """
        super().__init__()
    
        self.lr = lr
        
        self.numberOfAtoms = numberOfAtoms
        if pixelOutput == True:
            self.label_dims = 0
            label_size = grid_size*grid_size
            self.scaler = 1
        else:
            self.label_dims = 2
            label_size = self.label_dims*numberOfAtoms
            self.scaler = grid_size
        self.BFD_size = 38 #input size of 38x38 (BFD size at 100x100)
        self.example_input_array = torch.zeros((1, 2, self.BFD_size*self.BFD_size+2), device=device, requires_grad=True)

        self.prePreCNN = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.prePreCNN = torch.compile(self.prePreCNN)
        self.cnnOutputSize = 128 * (self.BFD_size // 8) * (self.BFD_size // 8)  
        self.obs_size = self.cnnOutputSize + 2  # +2 for x and y position
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
        self.preComputeNN = preComputeTransformer(obs_size=self.obs_size, numberOfHeads=self.nhead)
        # self.preComputeNN = torch.compile(self.preComputeNN)
        self.finalLayerNN = FinalLayer(self.obs_size, label_size) 
        # self.finalLayerNN = torch.compile(self.finalLayerNN)

        if pixelOutput: self.loss_fct = nn.BCEWithLogitsLoss()#nn.MSELoss()
        else: self.loss_fct = geomloss.SamplesLoss()

    def forward(self, ptychoImagesWithPositions: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the coordinates of the atoms.

        Args:
            ptychoImages: current Zernike moments

        Returns:
            coordinates of the atoms ordered by x position

        """
        B, T, D = ptychoImagesWithPositions.shape
        imageDim = self.BFD_size * self.BFD_size
        #shape: (batch_size, numberOfPositions, BFD_size* BFD_size)
        ptychoImages, coordinates = ptychoImagesWithPositions[:,:,:imageDim], ptychoImagesWithPositions[:,:,imageDim:]
        assert coordinates.shape[2] == 2, f"Coordinates should have shape (batch_size, numberOfPositions, 2) but got {coordinates.shape}"
        ptychoImages = ptychoImages.reshape((B*T , 1, self.BFD_size, self.BFD_size))
        encodedPtychoImages = self.prePreCNN(ptychoImages)  
        encodedPtychoImages = encodedPtychoImages.reshape((B, T, self.cnnOutputSize))  # Reshape to (B, T, cnnOutputSize)
        encodedPtychoImagesWithCoordinates = torch.cat((encodedPtychoImages, coordinates), dim=-1)  # Add coordinates to the end
        currentHiddenState = self.preComputeNN(encodedPtychoImagesWithCoordinates).flatten(start_dim=1)

        currentLabel :Tensor= self.finalLayerNN(currentHiddenState)
        # currentLabel : Tensor = self.finalLayer(ptychoImages.flatten(start_dim=1))

        labelOrdered = currentLabel
        if not pixelOutput:
            labelOrdered = currentLabel.flatten(start_dim=1).reshape((-1, self.numberOfAtoms, self.label_dims))
            sorted_indices = torch.argsort(labelOrdered[:, :, 0], dim=1)
            labelOrdered = torch.gather(labelOrdered, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, self.label_dims))*self.scaler

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
        ptychoImagesWithPositions, atomPositionsLabel, mask = batch
        # x: [B, T, H*W + metadata_dim]
        B, T, D = ptychoImagesWithPositions.shape

        image_dim = self.BFD_size * self.BFD_size 
        metadata_dim = 2

        image_flat = ptychoImagesWithPositions[:, :, :image_dim]          # [B, T, H*W]
        metadata = ptychoImagesWithPositions[:, :, image_dim:]            # [B, T, 2]
        assert metadata.shape[2] == metadata_dim

        # Masked flatten
        valid_positions_mask = ~mask.view(B, T)              # [B, T] for some reason the mask for the transformer is inverted and is true for invalid positions. So here it is inverted
        # valid_indices = valid_mask.nonzero(as_tuple=False)  # [N, 2] — (batch_idx, time_idx)

        # Extract valid image patches
        valid_images = image_flat[valid_positions_mask]     # [N, H*W]

        valid_images = valid_images.view(-1, 1, self.BFD_size , self.BFD_size )  # [N, 1, H, W]

        # CNN encoding
        encoded_imgs = self.prePreCNN(valid_images).reshape((-1, self.cnnOutputSize))     # [B, cnn_output_dim]

        # Insert CNN outputs back into [B, T, cnn_output_dim]
        img_features = torch.zeros((B, T, self.cnnOutputSize), device=ptychoImagesWithPositions.device)
        img_features[valid_positions_mask] = encoded_imgs

        # Concatenate with metadata
        encodedPtychoImagesWithCoordinates = torch.cat([img_features, metadata], dim=2) 
        currentHiddenState = self.preComputeNN(encodedPtychoImagesWithCoordinates, mask).flatten(start_dim=1)
        currentLabel :Tensor= self.finalLayerNN(currentHiddenState.flatten(start_dim=1))        

        # loss = torch.nn.MSELoss()(currentLabel.flatten(start_dim=1), atomPositionsLabel.flatten(start_dim=1)/grid_size)
        if pixelOutput: loss = self.loss_fct(torch.clip(currentLabel.flatten(start_dim=1),0,1), atomPositionsLabel.flatten(start_dim=1))
        else: 
            currentLabelReshaped = currentLabel.flatten(start_dim=1).reshape((B,-1,self.label_dims))
            atomPositionsLabelReshapedAndScaled = atomPositionsLabel.flatten(start_dim=1).reshape((B,-1,self.label_dims))/self.scaler
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
    
class ThreePartLightningVIT(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        numberOfAtoms = grid_size_in_A**2,
        hidden_size: int = 1024,
        num_layers = 5,
        fc_num_layers = 3
    ) -> None:
        """Basic Transformer+Linear Model.

        Args:
            lr: learning rate

        """
        super().__init__()
    
        self.lr = lr
        
        self.numberOfAtoms = numberOfAtoms
        if pixelOutput == True:
            self.label_dims = 0
            label_size = grid_size*grid_size
            self.scaler = 1
        else:
            self.label_dims = 2
            label_size = self.label_dims*numberOfAtoms
            self.scaler = grid_size
        self.BFD_size = 38 #input size of 38x38 (BFD size at 100x100)
        self.example_input_array = torch.zeros((1, 2, self.BFD_size*self.BFD_size+2), device=device, requires_grad=True)
        self.image_pixel_norm = nn.LayerNorm(self.BFD_size*self.BFD_size)
        self.prePrePatchEmb = PatchEncoderWithMeta(patch_size=19, emb_dim=125, image_size=self.BFD_size)  # 19x19 patches, 128-dimensional embeddings
        self.patchEmbOutputSize = 125 + 3 # 125 for patch embedding, 2 for coordinates, 1 for patch index
        self.obs_size = self.patchEmbOutputSize
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
        self.input_norm = nn.LayerNorm(self.obs_size)
        self.preComputeNN = preComputeTransformer(obs_size=self.obs_size, numberOfHeads=self.nhead, hidden_size = hidden_size, num_layers=num_layers, cls_like=True)
        # self.preComputeNN = torch.compile(self.preComputeNN)
        self.finalLayerNN = FinalLayer(self.obs_size, label_size, num_layers= fc_num_layers) 
        # self.finalLayerNN = torch.compile(self.finalLayerNN)
        
        if pixelOutput: self.loss_fct = nn.BCEWithLogitsLoss()#nn.MSELoss()
        else: self.loss_fct = geomloss.SamplesLoss()

    def forward(self, ptychoImagesWithPositions: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the coordinates of the atoms.

        Args:
            ptychoImages: current Zernike moments

        Returns:
            coordinates of the atoms ordered by x position

        """
        B, T, D = ptychoImagesWithPositions.shape
        imageDim = self.BFD_size * self.BFD_size
        #shape: (batch_size, numberOfPositions, BFD_size* BFD_size)
        ptychoImages, coordinates = ptychoImagesWithPositions[:,:,:imageDim], ptychoImagesWithPositions[:,:,imageDim:]
        assert coordinates.shape[2] == 2, f"Coordinates should have shape (batch_size, numberOfPositions, 2) but got {coordinates.shape}"
        ptychoImages = ptychoImages.reshape((B*T , self.BFD_size *  self.BFD_size))
        ptychoImages = self.image_pixel_norm(ptychoImages)
        coordinates = coordinates.reshape((B*T, 2))  # Reshape to (B*T, 2)
        encodedPtychoImages = self.prePrePatchEmb(ptychoImages, coordinates)  
        encodedPtychoImagesWithCoordinates = encodedPtychoImages.reshape((B, T * 4, self.patchEmbOutputSize))  # Reshape to (B, T * num_patches, patchEmbOutputSize)
        encodedPtychoImagesWithCoordinates = self.input_norm(encodedPtychoImagesWithCoordinates)  # Normalize the input features
        currentHiddenState = self.preComputeNN(encodedPtychoImagesWithCoordinates).flatten(start_dim=1)

        currentLabel :Tensor= self.finalLayerNN(currentHiddenState)
        # currentLabel : Tensor = self.finalLayer(ptychoImages.flatten(start_dim=1))

        labelOrdered = currentLabel
        if not pixelOutput:
            labelOrdered = currentLabel.flatten(start_dim=1).reshape((-1, self.numberOfAtoms, self.label_dims))
            sorted_indices = torch.argsort(labelOrdered[:, :, 0], dim=1)
            labelOrdered = torch.gather(labelOrdered, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, self.label_dims))*self.scaler

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
        ptychoImagesWithPositions, atomPositionsLabel, mask = batch
        # x: [B, T, H*W + metadata_dim]
        B, T, D = ptychoImagesWithPositions.shape

        image_dim = self.BFD_size * self.BFD_size 
        metadata_dim = 2

        image_flat = ptychoImagesWithPositions[:, :, :image_dim]          # [B, T, H*W]
        metadata = ptychoImagesWithPositions[:, :, image_dim:]            # [B, T, 2]
        assert metadata.shape[2] == metadata_dim

        # Masked flatten
        valid_positions_mask = ~mask.view(B, T)              # [B, T]
        # for some reason the mask for the transformer is inverted and is true for invalid positions. So here it is inverted
        # valid_indices = valid_mask.nonzero(as_tuple=False)  # [N, 2] — (batch_idx, time_idx)

        # Extract valid image patches
        valid_images = image_flat[valid_positions_mask]     # [N, H*W]

        valid_images = valid_images.view(-1, self.BFD_size * self.BFD_size )  # [N, H* W]
        valid_images = self.image_pixel_norm(valid_images)  # Normalize the images
        coordinates = metadata[valid_positions_mask].view((-1,2))        # [N, 2]
        # Patch encoding
        encoded_imgs = self.prePrePatchEmb(valid_images, coordinates).reshape((-1, self.patchEmbOutputSize))     # [B, patchEmbOutputSize]

        # Insert Patch outputs back into [B, T *num_patches, patchEmbOutputSize]
        img_features = torch.zeros((B, T*4, self.patchEmbOutputSize), device=ptychoImagesWithPositions.device)
        expanded_mask = valid_positions_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B,T*4)  # [B, T* 4] expanded to match number of patches
        img_features[expanded_mask] = encoded_imgs

        inverted_expanded_mask = ~expanded_mask  # Invert mask for transformer
        currentHiddenState = self.preComputeNN(img_features, inverted_expanded_mask).flatten(start_dim=1)
        currentLabel :Tensor= self.finalLayerNN(currentHiddenState.flatten(start_dim=1))        

        # loss = torch.nn.MSELoss()(currentLabel.flatten(start_dim=1), atomPositionsLabel.flatten(start_dim=1)/grid_size)
        if pixelOutput: loss = self.loss_fct(torch.clip(currentLabel.flatten(start_dim=1),0,1), atomPositionsLabel.flatten(start_dim=1))
        else: 
            currentLabelReshaped = currentLabel.flatten(start_dim=1).reshape((B,-1,self.label_dims))
            atomPositionsLabelReshapedAndScaled = atomPositionsLabel.flatten(start_dim=1).reshape((B,-1,self.label_dims))/self.scaler
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

