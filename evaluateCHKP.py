from lightningTrain import loadModel, lightnModelClass, evaluater, ptychographicDataLightning
import os
import lightning.pytorch as pl
import sys


resultVectorLength = 0 
numberOfOSAANSIMoments = 20	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1

DIMTILES = 11
DIMENSION = DIMTILES//3 + 1

version = "ZernikeNormal_0807_DistOverCubedDistOnlyDist"

if "Zernike" in version:
    numChannels=resultVectorLength * DIMENSION**2
    numLabels = 2
    modelName = "ZernikeNormal"
else:
    numChannels = DIMENSION**2
    numLabels = 2
    modelName = "FullPixelGridML"
    
print("load model")
model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)

if len(sys.argv) > 1:
    labelFile = sys.argv[1]
else:
    labelFile = "labels.csv"

epochAndStep = "epoch=31-step=52480"
checkpoint_path = os.path.join("checkpoints",f"{version}",f"{epochAndStep}.ckpt")
model = lightnModelClass.load_from_checkpoint(checkpoint_path = checkpoint_path, model = model)
lightnModel = lightnModelClass(model)

lightnDataLoader = ptychographicDataLightning(modelName, classifier = False, indicesToPredict = None, labelFile=labelFile)
lightnDataLoader.setup()

# trainer = pl.Trainer(accelerator='gpu')
# trainer.test(model, dataloaders=lightnDataLoader.test_dataloader())

evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, model.to('cuda'), indicesToPredict = None, modelName = modelName, version = f"evaluation_{version}_{epochAndStep}", classifier=False)