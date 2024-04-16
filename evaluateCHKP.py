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
modelName = "FullPixelGridML"
DIMTILES = 12
if modelName == "Zernike":
    
    pool = 3
    numChannels=resultVectorLength * DIMTILES**2
    numLabels = (DIMTILES**2)//3**2
else:
    numChannels = DIMTILES**2
    numLabels = 9

print("load model")
model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)

model = lightnModelClass.load_from_checkpoint(checkpoint_path = os.path.join("checkpoints",f"FullPixelGridML_relCNN2024","epoch=20-step=138747.ckpt"), model = model)
lightnModel = lightnModelClass(model)

lightnDataLoader = ptychographicDataLightning(modelName, classifier = False, indicesToPredict = None)
lightnDataLoader.setup()

trainer = pl.Trainer()
trainer.test(model, dataloaders=lightnDataLoader.test_dataloader())

evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, model, indicesToPredict = None, modelName = modelName, version = "evaluation", classifier=False)