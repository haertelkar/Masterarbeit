from random import randint
import h5py
import numpy as np
from tqdm import tqdm

numberOfPosMax = 100
obs_size = 0
numberOfOSAANSIMoments = 40
nonPredictedBorderInA = 3
windowSizeInA = 16
nonPredictedBorderInCoordinates = 5*nonPredictedBorderInA
Size = windowSizeInA * 5 + 2*nonPredictedBorderInCoordinates
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        obs_size += 1

obs_size += 3 # 2 for x and y position of the agent and one because otherwise its prime, bad for transformer


for testOrTrain in ["train", "test"]:
    maximumValues = None

    with h5py.File(f"measurements_{testOrTrain}/training_data.hdf5", "r") as f, h5py.File(f"measurements_{testOrTrain}/training_data_{numberOfPosMax}.hdf5", "w") as fOut:
        for key in tqdm(f.keys(),  desc = f"Going through measurements_{testOrTrain} and choosing random scan positions."):
            zernikeMoments =np.array( f[key])
            numberOfScans =randint(5, numberOfPosMax)
            randCoords : np.ndarray = np.random.permutation(Size*Size)[:numberOfScans]
            randXCoords = (randCoords % Size).astype(int) - nonPredictedBorderInA
            randYCoords = (randCoords / Size).astype(int) - nonPredictedBorderInA 
            padding = np.zeros_like(randXCoords)
            imageOrZernikeMoment = zernikeMoments.reshape((Size,Size,-1))[randXCoords,randYCoords].reshape((numberOfScans,-1))
            imageOrZernikeMomentsWithCoords = np.concatenate((imageOrZernikeMoment, np.stack([randXCoords - nonPredictedBorderInCoordinates, randYCoords - nonPredictedBorderInCoordinates, padding]).T), axis = 1)

            fOut.create_dataset(key, data = imageOrZernikeMomentsWithCoords, compression="lzf", shuffle = True, chunks = (1,obs_size))