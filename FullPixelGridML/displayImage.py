import h5py
from matplotlib import pyplot as plt
import numpy as np

with h5py.File('measurements_train/training_data.hdf5', 'r') as f:
    plt.imsave("exampleOutput.png",(np.array(f.get(list(f.keys())[0])).reshape(15,15,20,20))[0,0])    

