import h5py
from matplotlib import pyplot as plt
import numpy as np

with h5py.File('measurements_train/training_data.hdf5', 'r') as f:
    array = (np.array(f.get(list(f.keys())[0])).reshape(85,85,-1))
    imageDim = np.around(np.sqrt(array.shape[-1])).astype(int)
    image = array.reshape(85,85,imageDim,imageDim)[0,0]
    plt.imsave("exampleOutput.png",image)    

