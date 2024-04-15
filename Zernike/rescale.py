import h5py
import numpy as np
from tqdm import tqdm

#go through the hdf5 file and scale and shift the data. The vectors are all of the same length. Scale featurewise
for testOrTrain in ["test", "train"]:
    maximumValues = None
    with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r") as f:
        for key in tqdm(f.keys(), desc = f"Finding the maximum values to scale to one in measurements_{testOrTrain}."):
            zernikeImages = np.array(f[key][...]).astype(np.float32)
            if maximumValues is None:
                maximumValues = np.absolute(np.copy(zernikeImages))
                
            maximumValues = np.maximum(np.abs(maximumValues), np.abs(zernikeImages))

    assert(maximumValues is not None)

    with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r+") as f:
        for key in tqdm(f.keys(), desc =f"Scaling every feature seperately in measurements_{testOrTrain}."):
            zernikeImages = np.array(f[key][...]).astype(np.float32)
            f[key][...] = zernikeImages/maximumValues

    with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r+") as f:
        for key in tqdm(f.keys(), desc = "Testing every feature seperately for scaling."):
            if(not(np.all(np.array(f[key][...]).astype(np.float32) <= 1.1) and np.all(-1.1 < np.array(f[key][...]).astype(np.float32)))):
                raise Exception(f"Scaling unsuccesful: f[key] = {f[key][...]}")