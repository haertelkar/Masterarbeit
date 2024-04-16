import time

from tqdm import tqdm
from Zernike.ZernikeTransformer import Zernike, calc_diameter_bfd


numberOfOSAANSIMoments = 20
ZernikeObject = Zernike(maxR = 36, numberOfOSAANSIMoments= numberOfOSAANSIMoments)

#create a numpy array of size (12,12,110,110) filled with random number
import numpy as np
images = np.random.rand(12*12,110,110)


print("measuring with only nonzero elements\n")

startTime = time.time()
for i in range(10):
    zernikeValues1 = ZernikeObject.zernikeTransform(fileName = None, groupOfPatterns = images, zernikeTotalImages = None)
    
print("Time taken for 10 iterations: ", time.time() - startTime, " seconds")

ZernikeObject.dimToBasis = {}
startTime = time.time()
for i in range(10):
    zernikeValues2 = ZernikeObject.zernikeTransformVectorized(fileName = None, groupOfPatterns = images, zernikeTotalImages = None)
print("Time taken for 10 iterations in the numpy vectorized variant: ", time.time() - startTime, " seconds")

print("\n\nmeasuring with 50% nonzero elements\n")

images[np.random.choice(images.shape[0], int(0.5*images.shape[0]), replace = False),:,:] = np.zeros((images.shape[-2],images.shape[-1]))
ZernikeObject.dimToBasis = {}
startTime = time.time()
for i in range(100):
    zernikeValues = ZernikeObject.zernikeTransform(fileName = None, groupOfPatterns = images, zernikeTotalImages = None)
print("Time taken for 10 iterations: ", time.time() - startTime, " seconds")

ZernikeObject.dimToBasis = {}
startTime = time.time()
for i in range(100):
    zernikeValues = ZernikeObject.zernikeTransformVectorized(fileName = None, groupOfPatterns = images, zernikeTotalImages = None)
print("Time taken for 10 iterations in the numpy vectorized variant: ", time.time() - startTime, " seconds")