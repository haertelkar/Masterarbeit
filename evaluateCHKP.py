from lightningTrain import loadModel, lightnModelClass, evaluater, ptychographicDataLightning
import os
import lightning.pytorch as pl
import sys
from tqdm import tqdm

resultVectorLength = 0 
numberOfOSAANSIMoments = 20	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1

DIMTILES = 10
DIMENSION = DIMTILES//3 + 1


if len(sys.argv) > 1:
    inVersion = sys.argv[1]
else:
    inVersion = ""

for folder in tqdm(os.listdir("checkpoints")):
    if inVersion not in folder:
        continue
    version = folder
    for file in os.listdir(os.path.join("checkpoints", folder)):
        if not file.endswith(".ckpt"):
            continue

        if "Zernike" in version:
            numChannels=resultVectorLength * DIMENSION**2
            modelName = "ZernikeNormal"
        else:
            numChannels = DIMENSION**2
            modelName = "FullPixelGridML"
            
        if "_atom" in version:
            atomNo = version[-1]
            if "onlyDist" in version:
                numLabels = 2
                labelFile = f"labels_only_Dist_{atomNo}.csv"
            elif "onlyElem" in version:
                numLabels = 1
                labelFile = f"labels_only_Elem_{atomNo}.csv"
            else:
                tqdm.write(f"Skipping {version} as it is not implemented")
                continue
        else:
            if "onlyDist" in version:
                numLabels = 4 * 2
                labelFile = "labels_only_Dist.csv"
            elif "onlyElem" in version:
                numLabels = 4
                labelFile = "labels_only_Elem.csv"
            else:
                numLabels = 4 * 3
                labelFile = "labels.csv"

        model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)


        epochAndStep = file
        tqdm.write(f"Evaluating {epochAndStep} of {version}")
        checkpoint_path = os.path.join("checkpoints",f"{version}",f"{epochAndStep}")
        model = lightnModelClass.load_from_checkpoint(checkpoint_path = checkpoint_path, model = model)
        lightnModel = lightnModelClass(model)

        lightnDataLoader = ptychographicDataLightning(modelName, labelFile=labelFile, testDirectory="measurements_test", onlyTest=True)
        lightnDataLoader.setup()

        # trainer = pl.Trainer(accelerator='gpu')
        # trainer.test(model, dataloaders=lightnDataLoader.test_dataloader())

        evaluater(lightnDataLoader.test_dataloader(), lightnDataLoader.test_dataset, model.to('cuda'), indicesToPredict = None, modelName = modelName, version = f"evaluation_{version}_{epochAndStep}", classifier=False)