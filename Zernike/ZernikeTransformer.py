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
    from Zernike.ZernikePolynomials import Zernike, seperateFileNameAndCoords
from mpi4py import MPI
import h5py
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
worldsize = comm.Get_size()

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
        with h5py.File(os.path.join(imagePath(testOrTrain), "training_data.hdf5"), 'r') as totalImages, h5py.File(os.path.join(f"measurements_{testOrTrain}", f"training_data_{rank}.hdf5"), 'w') as zernikeTotalImages:
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
                image = np.array(totalImages[fileName])[xCoord, yCoord]
                if ZernikeObject is None:
                    radius = radius or int(len(image)/2)
                    ZernikeObject = Zernike(radius, image.shape[-1], noOfMoments)
                    
                for cnt, row in enumerate(Reader):
                    fileNameWithCoords = row[0]
                    _, _, fileName = seperateFileNameAndCoords(fileNameWithCoords)
                    imageFileNames.add(fileName)

            if rank == 0: shutil.copy(os.path.join(imgPath, "labels.csv"), os.path.join(f"measurements_{testOrTrain}", "labels.csv"))
            shutil.copy(os.path.join(imgPath, "labels.csv"), os.path.join(f"measurements_{testOrTrain}", "labels.csv"))
            for cnt, fileName in enumerate(tqdm(imageFileNames, desc= f"Going through files in measurements_{testOrTrain}", total = len(imageFileNames), leave = leave)):
                if cnt%worldsize == rank:
                    images = np.array(totalImages[fileName])
                    ZernikeObject.zernikeTransform(fileName, images, zernikeTotalImages)

    os.chdir(oldDir)

if __name__ == "__main__":
    zernikeTransformation()