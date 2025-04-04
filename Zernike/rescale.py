import h5py
import numpy as np
from tqdm import tqdm
import glob
# print("[INFO] This script is not necessary.")
# exit()

#go through the hdf5 file and scale the data. The vectors are all of the same length. Scale featurewise
for testOrTrain in ["test", "train"]:
    maximumValues = None
    for hdf5file in tqdm(glob.glob(f"measurements_{testOrTrain}/*.hdf5"), desc = f"Finding the maximum values to scale to one in measurements_{testOrTrain}."):
        try:
            with h5py.File(f"{hdf5file}", "r") as f:
                for key in tqdm(list(f.keys()), desc = f"Finding the maximum values to scale to one in {hdf5file}.", leave=False):
                    zernikeImages = np.array(f[key][...]).astype(np.float32)
                    for zernikeImage in zernikeImages:
                        if zernikeImage is None:
                            raise Exception(f"zernikeImages is None: {hdf5file}{key}")
                        if maximumValues is None:
                            maximumValues = np.absolute(np.copy(zernikeImage))
                            
                        maximumValues = np.maximum(np.abs(maximumValues), np.abs(zernikeImage))
        except BlockingIOError:
            continue


    assert(maximumValues is not None)

    # with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r+") as f:
    #     for key in tqdm(f.keys(), desc =f"Scaling every feature seperately in measurements_{testOrTrain}."):
    #         zernikeImages = np.array(f[key][...]).astype(np.float32)
    #         f[key][...] = zernikeImages/maximumValues

    with open("scalingFactorsNew.csv", "w") as f:
        for scalar in maximumValues:
            f.write(f"{scalar},")

    #with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r+") as f:
    #    for key in tqdm(f.keys(), desc = "Testing every feature seperately for scaling."):
     #       if(not(np.all(np.array(f[key][...]).astype(np.float32) <= 1.1) and np.all(-1.1 < np.array(f[key][...]).astype(np.float32)))):
     #           raise Exception(f"Scaling unsuccessful: f[key] = {f[key][...]}")