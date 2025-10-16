import h5py
from matplotlib import pyplot as plt

with h5py.File('/data/scratch/haertelk/Masterarbeit/FullPixelGridML/measurements_test_4to8s_-50def_15B_new/119_17517350982703.hdf5', 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    
    # Get the first object name
    a_group_key = list(f.keys())[0]
    
    # Get the object type
    a_dataset = f[a_group_key]
    
    # Print the shape and datatype of the dataset
    print("Dataset shape: %s" % str(a_dataset.shape))
    print("Dataset datatype: %s" % str(a_dataset.dtype))
    
    # Read the data
    data = a_dataset[:]
    firstData = data[0]
    coordinates = firstData[38*38:]
    print("Coordinates: %s" % str(coordinates))
    firstImage = firstData[:38*38].reshape(38, 38)
    plt.imsave('firstImage.png', firstImage, cmap='gray')
    