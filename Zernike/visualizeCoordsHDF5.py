import h5py
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_coords_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        x_coords = []
        y_coords = []
        for name, dataset in tqdm(f.items()):
            data = dataset[:]
            for position in data:
                if len(position) == 864:
                    x = position[-3]
                    y = position[-2]
                    if -35 <= x <= 49 and -35 <= y <= 49:
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        print(f"Invalid coordinates in dataset {name}: x={x}, y={y}")
                else:
                    print(f"Unexpected data length in dataset {name}: {len(position)}")

        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(85, 85), range=[[-35, 49], [-35, 49]])
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
        plt.colorbar()
        plt.title('Heatmap of Coordinates')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig("heatmap_hdf5_coords.png")

# Replace 'your_file.hdf5' with the path to your HDF5 file
visualize_coords_hdf5('/data/scratch/haertelk/Masterarbeit/Zernike/measurements_train/training_data_100_Grid.hdf5')