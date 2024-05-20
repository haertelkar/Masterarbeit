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

DIMTILES = 12

version = "ZernikeNormal_1905_onlyGridPos"

if "Zernike" in version:
    pool = 3
    numChannels=resultVectorLength * DIMTILES**2
    numLabels = 9
    modelName = "ZernikeNormal"
else:
    numChannels = DIMTILES**2
    numLabels = (DIMTILES**2)//3**2
    modelName = "FullPixelGridML"
    
print("load model")
model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)

epochAndStep = "epoch=14-step=12585"
checkpoint_path = os.path.join("checkpoints",f"{version}",f"{epochAndStep}.ckpt")
model = lightnModelClass.load_from_checkpoint(checkpoint_path = checkpoint_path, model = model)
lightnModel = lightnModelClass(model)

lightnDataLoader = ptychographicDataLightning(modelName, classifier = False, indicesToPredict = None)
lightnDataLoader.setup()

trainer = pl.Trainer(accelerator='gpu')
trainer.test(model, dataloaders=lightnDataLoader.test_dataloader())

evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, model, indicesToPredict = None, modelName = modelName, version = f"evaluation_{version}_{epochAndStep}", classifier=False)