import h5py
import numpy as np
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
# print("[INFO] This script is not necessary.")
# exit()
# zernikeImages0 = []
# zernikeImages400 = []
# zernikeImages800 = []
zernikeImagesAll = []
#go through the hdf5 file and scale the data. The vectors are all of the same length. Scale featurewise
for testOrTrain in ["train"]:
    meanValues = None
    with h5py.File(f"measurements_train_1710_4to8s_-50def_15B_860Z_OSA/training_data.hdf5", "r") as f:
        for key in tqdm(list(f.keys())[::100], desc = f"Finding the maximum values to scale to one in measurements_train_1710_4to8s_-50def_15B_860Z_OSA/training_data.hdf5."):
            zernikeImages = np.array(f[key][...]).astype(np.float32)
            
            # zernikeImages0.append(zernikeImages[:,0])
            # zernikeImages400.append(zernikeImages[:,400])
            # zernikeImages800.append(zernikeImages[:,800])
            # tqdm.write(f"{hdf5file}: {str(zernikeImages[0,800])}, {zernikeImages[99,800]}")
            for image in zernikeImages:
                zernikeImagesAll.append(image)


zernikeImagesAll = np.array(zernikeImagesAll)
meanValues = np.mean(zernikeImagesAll, axis=0)
print("meanValues.shape", meanValues.shape)
stdValues = np.std(zernikeImagesAll, axis=0)
print("stdValues.shape", stdValues.shape)
# zernikeImages0 = np.array(zernikeImages0)
# zernikeImages400 = np.array(zernikeImages400)
# zernikeImages800 = np.array(zernikeImages800)

# # Plotting the histograms
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.hist(zernikeImages0.flatten(), bins=100, color='blue', alpha=0.7)
# plt.title('Histogram of Zernike Images (0)')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.subplot(3, 1, 2)
# plt.hist(zernikeImages400.flatten(), bins=100, color='green', alpha=0.7)
# plt.title('Histogram of Zernike Images (400)')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.subplot(3, 1, 3)
# plt.hist(zernikeImages800.flatten(), bins=100, color='red', alpha=0.7)
# plt.title('Histogram of Zernike Images (800)')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig(f"histogram_{testOrTrain}.png")
# plt.close()

# assert(meanValues is not None)

# with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r+") as f:
#     for key in tqdm(f.keys(), desc =f"Scaling every feature seperately in measurements_{testOrTrain}."):
#         zernikeImages = np.array(f[key][...]).astype(np.float32)
#         f[key][...] = zernikeImages/meanValues

with open("meanValues.csv", "w") as f:
    for scalar in meanValues:
        f.write(f"{scalar},")
with open("stdValues.csv", "w") as f:
    for scalar in stdValues:
        f.write(f"{scalar},")

    #with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r+") as f:
    #    for key in tqdm(f.keys(), desc = "Testing every feature seperately for scaling."):
     #       if(not(np.all(np.array(f[key][...]).astype(np.float32) <= 1.1) and np.all(-1.1 < np.array(f[key][...]).astype(np.float32)))):
     #           raise Exception(f"Scaling unsuccessful: f[key] = {f[key][...]}")