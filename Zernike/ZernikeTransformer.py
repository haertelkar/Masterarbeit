import re
import numpy as np
import os
import csv
from tqdm import tqdm
from mahotas.features import zernike_moments
import matplotlib.pyplot as plt
import shutil
from joblib import Parallel, delayed
try:
    from ZernikePolynomials import Zernike, seperateFileNameAndCoords
except ModuleNotFoundError:
    from Zernike.ZernikePolynomials import Zernike
# from mpi4py import MPI
import h5py

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# worldsize = comm.Get_size()

def imagePath(testOrTrain):
    return os.path.join("..","FullPixelGridML",f"measurements_{testOrTrain}")

def splitForEachRank(arr, size):
    arr = np.array(list(arr))
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr   = arr[size:]
    arrs.append(arr)
    return arrs

def zernikeTransformation(pathToZernikeFolder = os.getcwd(), radius = 15, noOfMoments = 40, leave = True):
    oldDir = os.getcwd() 
    os.chdir(pathToZernikeFolder)
    ZernikeObject = None

    for testOrTrain in ["train", "test"]:
        imgPath = imagePath(testOrTrain)
        imageFileNames = set()
        with open(os.path.join(imgPath, 'labels.csv'), 'r', newline='') as labelsFullPixelGrid:
            Reader = csv.reader(labelsFullPixelGrid, delimiter=',', quotechar='|')
            for cnt, row in enumerate(Reader):
                if cnt == 0: continue #skips the header line 
                firstEntry = row #reads out first row
                break
            fileNameWithCoords = firstEntry[0]
            xCoord, yCoord, fileName = seperateFileNameAndCoords(fileNameWithCoords)
            imageFileNames.add(fileName)
            if ".npy" not in fileName: raise Exception(f"{fileName} is not a valid filename")
            image = np.load(os.path.join(imgPath, fileName))[xCoord, yCoord]
            if ZernikeObject is None:
                radius = radius or int(len(image)/2)
                ZernikeObject = Zernike(radius, image.shape[-1], noOfMoments)
                
            for cnt, row in enumerate(Reader):
                fileNameWithCoords = row[0]
                if ".npy" not in fileNameWithCoords: raise Exception(f"{fileNameWithCoords} is not a valid filename")
                _, _, fileName = seperateFileNameAndCoords(fileNameWithCoords)
                imageFileNames.add(fileName)

        shutil.copy(os.path.join(imgPath, "labels.csv"), os.path.join(f"measurements_{testOrTrain}", "labels.csv"))
        with h5py.File(os.path.join(imagePath(testOrTrain), "training_data.hdf5"), 'r') as totalImages, h5py.File(os.path.join(f"measurements_{testOrTrain}", "training_data.hdf5"), 'w') as zernikeTotalImages:
            for fileName in enumerate(tqdm(imageFileNames, desc= f"Going through files in measurements_{testOrTrain}", total = len(imageFileNames), leave = leave)):
                images = np.array(totalImages[fileName])
                ZernikeObject.zernikeTransform(testOrTrain, fileName, images, zernikeTotalImages)
                
    os.chdir(oldDir)

if __name__ == "__main__":
    zernikeTransformation()