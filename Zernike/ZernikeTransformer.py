import gc
import glob
import re
import resource
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

def findUnusedFileName(path, fileNameNoModifications:str) -> str:
    fileNameModified = f"{fileNameNoModifications}"
    while(os.path.exists(os.path.join(path,f"{fileNameModified}.hdf5"))):
        print(fileNameModified, "taken")
        fileNameModified = fileNameModified + "x"
    print(fileNameModified, "chosen")
    return f"{fileNameModified}"

def readProgressAcrossAllRuns(path:str):
    fileNamesDone = set()
    for progressFile in glob.glob(os.path.join(path, "*.progress")):
        with open(progressFile) as pF:
            for fileName in pF:
                fileNamesDone.add(fileName)

    return fileNamesDone


def zernikeTransformation(pathToZernikeFolder = os.getcwd(), radius = 15, noOfMoments = 40, leave = True):
    oldDir = os.getcwd() 
    os.chdir(pathToZernikeFolder)
    ZernikeObject = None
    for testOrTrain in ["train", "test"]:
        zernikeFileName = findUnusedFileName(f"measurements_{testOrTrain}", f"training_data_{rank}")
        zernikeHdf5FileName = zernikeFileName + ".hdf5"
        zernikeProgressFileName = zernikeFileName + ".progress"
        fileNamesDone = readProgressAcrossAllRuns(f"measurements_{testOrTrain}")
        comm.barrier()

        with h5py.File(os.path.join(f"measurements_{testOrTrain}", zernikeHdf5FileName), 'w') as zernikeTotalImages:
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
                with h5py.File(os.path.join(imagePath(testOrTrain), "training_data.hdf5"), 'r') as totalImages:
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
            totalNumberOfFiles = len(imageFileNames)
            imageFileNames = imageFileNames.difference(fileNamesDone)
            for cnt, fileName in enumerate(tqdm(imageFileNames, desc= f"Going through files in measurements_{testOrTrain}", total = totalNumberOfFiles//worldsize + (totalNumberOfFiles%worldsize > 0), initial=len(fileNamesDone)//worldsize + (fileNamesDone%worldsize > 0), leave = leave, disable = (rank != 0))):
                if cnt%worldsize == rank:
                    tqdm.write(f"rank: {rank}\nfileName: {fileName} \nram usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                    with h5py.File(os.path.join(imagePath(testOrTrain), "training_data.hdf5"), 'r') as totalImages:
                        images = np.array(totalImages[fileName])
                    ZernikeObject.zernikeTransform(fileName, images, zernikeTotalImages)
                    images = None
                    gc.collect()
                    with open(os.path.join(f"measurements_{testOrTrain}",zernikeProgressFileName), "a+") as progressFile:
                        progressFile.write(f"{fileName}\n")


    os.chdir(oldDir)

if __name__ == "__main__":
    zernikeTransformation()