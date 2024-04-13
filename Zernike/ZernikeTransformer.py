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
    from ZernikePolynomials import Zernike
except ModuleNotFoundError:
    from Zernike.ZernikePolynomials import Zernike
from mpi4py import MPI
import h5py
import sys


def seperateFileNameAndCoords(fileNameAndCoords : str):
    try:
        fileName, xCoord, yCoord = fileNameAndCoords.split(r"[")
    except AttributeError as e:
        raise Exception(f"{e}\n The entry {fileNameAndCoords} could not be seperated in fileName, xCoord, yCoord")
    except ValueError as e:
        raise Exception(f"{e}\n The entry {fileNameAndCoords} could not be seperated in fileName, xCoord, yCoord")
    xCoord = int(xCoord[:-1])
    yCoord = int(yCoord[:-1])
    return xCoord, yCoord, fileName

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


def zernikeTransformation(pathToZernikeFolder = os.getcwd(), radius = 0, noOfMoments = 20, leave = True): #radius set below & noOfMoment = 10 tested works, noOfMoment = 40 optimal performance
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
            if rank == 0:
                with open(os.path.join(imgPath, 'labels.csv'), 'r', newline='') as labelsFullPixelGrid:
                    Reader = csv.reader(labelsFullPixelGrid, delimiter=',', quotechar='|')
                    for cnt, row in enumerate(Reader):
                        if cnt == 0: continue #skips the header line 
                        imageFileNames.add(row[0])

                shutil.copy(os.path.join(imgPath, "labels.csv"), os.path.join(f"measurements_{testOrTrain}", "labels.csv"))
                imageFileNames = imageFileNames.difference(fileNamesDone)
                imageFileNames = list(imageFileNames)
            else:
                imageFileNames = None
            imageFileNames = comm.bcast(imageFileNames, root=0)
            totalNumberOfFiles = len(imageFileNames)
            with h5py.File(os.path.join(imagePath(testOrTrain), "training_data.hdf5"), 'r') as totalImages:
                randomFileName = imageFileNames[0]
                randomGroupOfPatterns = np.array(totalImages[randomFileName]) #this image always exists so it is easy to just use it
                imageDim = randomGroupOfPatterns.shape[-1] 
                randomImage = randomGroupOfPatterns[0]
            if ZernikeObject is None:
                diameterBFD = calc_diameter_bfd(randomImage)
                radius = diameterBFD//2+1
                ZernikeObject = Zernike(radius, noOfMoments)
            for cnt, fileName in enumerate(tqdm(imageFileNames, desc= f"Going through files in measurements_{testOrTrain}", total = totalNumberOfFiles, initial=len(fileNamesDone), leave = leave, disable = (rank != 0))):
                if cnt%worldsize != rank:
                    continue
                #tqdm.write(f"rank: {rank}\nfileName: {fileName} \nram usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                with h5py.File(os.path.join(imagePath(testOrTrain), "training_data.hdf5"), 'r') as totalImages:
                    groupOfPatterns = np.array(totalImages[fileName])
                ZernikeObject.zernikeTransform(fileName, groupOfPatterns, zernikeTotalImages)
                groupOfPatterns = None
                gc.collect()
                with open(os.path.join(f"measurements_{testOrTrain}",zernikeProgressFileName), "a+") as progressFile:
                    progressFile.write(f"{fileName}\n")

    os.chdir(oldDir)
    comm.barrier()
    print("Finished Zernike Transformations.")

def calc_diameter_bfd(image):
    brightFieldDisk = np.zeros_like(image)
    brightFieldDisk[image > np.max(image)*0.05] = 1
    bfdArea = np.sum(brightFieldDisk)
    diameterBFD = np.sqrt(bfdArea/np.pi) * 2
    return diameterBFD

if __name__ == "__main__":
    zernikeTransformation()