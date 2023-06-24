from CleanUp import cleanUp
from Zernike.ZernikeTransformer import zernikeTransformation
from train import Learner
from tqdm import tqdm
import sys, os

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

radii = [10,15,20] #zero is full radius
orders = [300,350,400,450]
radiiAndOrders = []

for radius in radii:
    for order in orders:
        radiiAndOrders.append([radius, order])

for radius, order in tqdm(radiiAndOrders, desc="Calculating different radii and up to different orders"): 
    cleanUp("Zernike")
    blockPrint()
    zernikeTransformation("Zernike",radius = radius, noOfMoments= order, leave=False)
    learn = Learner(epochs = 300, version = f"radius{radius or 'Full'}_order{order}", classifier=True, indicesToPredict=1)

    learn.Learner(modelName="Zernike", leave = False)
    enablePrint()

        
