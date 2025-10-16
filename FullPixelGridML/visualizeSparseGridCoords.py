from random import randint

from tqdm import tqdm
from SimulateTilesOneFile import generate_sparse_grid
import matplotlib.pyplot as plt

x_coords = []
y_coords = []
import numpy as np
nonPredictedBorderInCoords = 15
windowLengthinCoords = 15
for cnt in tqdm(range(10000)):
    sparseGridFactor = 4
    MaxShift = nonPredictedBorderInCoords + windowLengthinCoords//2  
    xShift = randint(-MaxShift, MaxShift)
    yShift = randint(-MaxShift, MaxShift)
    xStartShift = 0#max(0, xShift)
    yStartShift = 0#max(0, yShift)
    xEndShift = 0#min(xShift, 0 )
    yEndShift = 0#min(yShift, 0)
    xOffset = randint(0,sparseGridFactor-1)
    yOffset = randint(0,sparseGridFactor-1)
    choosenCoords2d : np.ndarray = np.array(generate_sparse_grid(nonPredictedBorderInCoords * 2 + windowLengthinCoords,nonPredictedBorderInCoords * 2 + windowLengthinCoords,
                                                                        sparseGridFactor, xStartShift=xStartShift, yStartShift=yStartShift,
                                                                        xEndShift=xEndShift,yEndShift=yEndShift, xOffset = xOffset, 
                                                                        yOffset = yOffset, twoD = True)) - nonPredictedBorderInCoords
        
    
    x_coords_part, y_coords_part = choosenCoords2d[:, 0], choosenCoords2d[:, 1]
    x_coords.extend(x_coords_part)
    y_coords.extend(y_coords_part)
heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(85, 85), range=[[-35, 49], [-35, 49]])
extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
plt.colorbar()
plt.title('Heatmap of Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.savefig("heatmap_hdf5_coords.png")