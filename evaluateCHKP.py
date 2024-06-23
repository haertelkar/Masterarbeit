from lightningTrain import loadModel, lightnModelClass, evaluater, ptychographicDataLightning
import os
import lightning.pytorch as pl


resultVectorLength = 0 
numberOfOSAANSIMoments = 20	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1

DIMTILES = 11
DIMENSION = DIMTILES//2 + 1

version = "ZernikeNormal_1806_newDataDistFix_20n"

if "Zernike" in version:
    numChannels=resultVectorLength * DIMENSION**2
    numLabels = 9
    modelName = "ZernikeNormal"
else:
    numChannels = DIMENSION**2
    numLabels = 9
    modelName = "FullPixelGridML"
    
print("load model")
model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)

epochAndStep = "epoch=13-step=784"
checkpoint_path = os.path.join("checkpoints",f"{version}",f"{epochAndStep}.ckpt")
model = lightnModelClass.load_from_checkpoint(checkpoint_path = checkpoint_path, model = model)
lightnModel = lightnModelClass(model)

lightnDataLoader = ptychographicDataLightning(modelName, classifier = False, indicesToPredict = None)
lightnDataLoader.setup()

trainer = pl.Trainer(accelerator='gpu')
# trainer.test(model, dataloaders=lightnDataLoader.test_dataloader())

evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, model, indicesToPredict = None, modelName = modelName, version = f"evaluation_{version}_{epochAndStep}", classifier=False)