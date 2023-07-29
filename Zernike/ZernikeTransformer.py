import numpy as np
import os
import csv
from tqdm import tqdm
from mahotas.features import zernike_moments
import matplotlib.pyplot as plt
import shutil
from joblib import Parallel, delayed
try:
    from ZernikePolynomials import Zernike, imagePath
except ModuleNotFoundError:
    from Zernike.ZernikePolynomials import Zernike

def zernikeTransformation(pathToZernikeFolder = os.getcwd(), radius = 15, noOfMoments = 40, leave = True):
    oldDir = os.getcwd() 
    os.chdir(pathToZernikeFolder)
    ZernikeObject = None

    for testOrTrain in ["train", "test"]:
        imgPath = imagePath(testOrTrain)
        imageFileNames = []
        with open(os.path.join(imgPath, 'labels.csv'), 'r', newline='') as labelsFullPixelGrid:
            Reader = csv.reader(labelsFullPixelGrid, delimiter=',', quotechar='|')
            for cnt, row in enumerate(Reader):
                if cnt == 0: continue #skips the header line 
                firstEntry = row #reads out first row
                if cnt == 2: break
            fileName = firstEntry[0]
            if ".npy" not in fileName: raise Exception(f"{fileName} is not a valid filename")
            image = np.load(os.path.join(imgPath, fileName))
            if ZernikeObject is None:
                radius = radius or int(len(image)/2)
                ZernikeObject = Zernike(radius, image.shape[-1], noOfMoments)
                
            for cnt, row in enumerate(Reader):
                fileName = row[0]
                if ".npy" not in fileName: raise Exception(f"{fileName} is not a valid filename")
                imageFileNames.append(fileName)

        shutil.copy(os.path.join(imgPath, "labels.csv"), os.path.join(f"measurements_{testOrTrain}", "labels.csv"))
        # for fileName in tqdm(imageFileNames, desc= "Going through files", total = len(imageFileNames), leave = leave):
        #     ZernikeObject.zernikeTransform(testOrTrain, fileName)
        Parallel(n_jobs=20)(delayed(ZernikeObject.zernikeTransform)(testOrTrain, fileName) for fileName in imageFileNames)

    os.chdir(oldDir)

if __name__ == "__main__":
    zernikeTransformation()