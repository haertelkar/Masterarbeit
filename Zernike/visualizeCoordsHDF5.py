import h5py
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_coords_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        x_coords = []
        y_coords = []
        for cnt, (name, dataset) in enumerate(tqdm(f.items())):
            if not cnt % 50 == 0:
                continue
            try:
                data = f[name][:]
            except Exception as e:
                print(f"Error reading dataset {name}: {e}")
                continue
            for cnt, position in enumerate(data):
                # if len(position) == 864:
                x = position[-2]
                y = position[-1]
                if -35 <= x <= 49 and -35 <= y <= 49:
                    x_coords.append(x)
                    y_coords.append(y)
                else:
                    print(f"Invalid coordinates in dataset {name}: x={x}, y={y}")
                # else:
                #     print(f"Unexpected data length in dataset {name}: {len(position)}")
        # if cnt != 99: 
        #     print(f"Warning: Expected 100 datasets, but found {cnt + 1} datasets.")
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(85, 85), range=[[-35, 49], [-35, 49]])
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
        plt.colorbar()
        plt.title('Heatmap of Coordinates')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig("heatmap_hdf5_coords.png")

# Replace 'your_file.hdf5' with the path to your HDF5 file
visualize_coords_hdf5('/data/scratch/haertelk/Masterarbeit/Zernike/measurements_test_4sparse_0def_0B_new_20Z/training_data.hdf5')
# visualize_coords_hdf5('/data/scratch/haertelk/Masterarbeit/Zernike/measurements_train/3429321_17473813309959483.hdf5')