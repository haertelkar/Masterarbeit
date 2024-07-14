import glob
from typing import Tuple
from ase import Atoms
from ase.visualize import view
from ase.io import read
from ase.build import surface, mx2, graphene
from ase.geometry import get_distances
# from abtem.potentials import Potential
import matplotlib.pyplot as plt
from abtem.waves import Probe
import numpy as np
from abtem.measure import block_zeroth_order_spot, Measurement
from abtem.scan import GridScan
from abtem.detect import PixelatedDetector
from random import random, randint, choice, uniform
from abtem.reconstruct import MultislicePtychographicOperator, RegularizedPtychographicOperator
from abtem import Potential, FrozenPhonons, Probe, CTF
from abtem.detect import AnnularDetector, PixelatedDetector
from abtem.scan import GridScan
from abtem.noise import poisson_noise
from abtem.structures import orthogonalize_cell
import torch
from tqdm import tqdm
import csv
import cv2
import warnings
import os
from numba import njit
from ase import Atoms
from ase.build import molecule, bulk, surface
from time import time
import faulthandler
import signal
from itertools import combinations
import h5py
from ase.build import nanotube
from skimage.measure import block_reduce
faulthandler.register(signal.SIGUSR1.value)
from mp_api.client import MPRester
# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring(seconds_frozen=60, test_interval=100)


device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Calculating on {device}")
xlen = ylen = 5

def calc_diameter_bfd(image):
    brightFieldDisk = np.zeros_like(image)
    brightFieldDisk[image > np.max(image)*0.1 + np.min(image)*0.9] = 1
    bfdArea = np.sum(brightFieldDisk)
    diameterBFD = np.sqrt(bfdArea/np.pi) * 2
    return diameterBFD

def moveAndRotateAtomsAndOrthogonalize(atoms, xPos = None, yPos = None, zPos = None, ortho = True) -> Atoms:
    xPos = random()*xlen/3 + xlen/3 if xPos is None else xPos
    yPos = random()*ylen/3 + ylen/3 if yPos is None else yPos
    zPos = 0 if zPos is None else zPos
    atoms.positions += np.array([xPos, yPos, zPos])[None,:]    
    atoms.rotate("x", randint(0,360))
    atoms.rotate("y", randint(0,360))
    atoms.rotate("z", randint(0,360))
    if ortho: atoms = orthogonalize_cell(atoms, max_repetitions=10)
    return atoms # type: ignore

def createAtomPillar(xPos = None, yPos = None, zPos = None, zAtoms = randint(1,10), xAtomShift = 0, yAtomShift = 0, element = None) -> Atoms:
    kindsOfElements = {6:0, 14:1, 74:2}
    element = choice(list(kindsOfElements.keys())) if element is None else element
    # maxShiftFactorPerLayer = 0.01
    #(random()  - 1/2) * maxShiftFactorPerLayer * xlen  #slant of atom pillar
    #(random() -  1/2) * maxShiftFactorPerLayer * ylen #slant of atom pillar
    # xAtomPillarAngle = np.arcsin(xAtomShift/1)
    # yAtomPillarAngle = np.arcsin(yAtomShift/1)
    positions = [np.array([0, 0, 0])]
    for atomBefore in range(zAtoms-1):
        positions += [positions[atomBefore] + np.array([xAtomShift, yAtomShift, 1])]
    atomPillar = Atoms(numbers = [element] * zAtoms , positions=positions, cell = [xlen, ylen, zAtoms,90,90,90])
    atomPillar_101 = surface(atomPillar, indices=(1, 0, 1), layers=2, periodic=True)
    atomPillar_slab = moveAndRotateAtomsAndOrthogonalize(atomPillar_101, xPos, yPos, zPos, ortho=True)
    return atomPillar_slab

#BROKEN #TODO use poisson disk
def multiPillars(xPos = None, yPos = None, zPos = None, zAtoms = randint(1,10), xAtomShift = 0, yAtomShift = 0, element = None, numberOfRandomAtomPillars = 3) -> Atoms:
    allPositions = []
    xPos = xPos if xPos is not None else -3
    yPos = yPos if yPos is not None else -3
    zPos = zPos if zPos is not None else 0
    atomPillar = createAtomPillar(xPos = xPos, yPos = yPos, zPos = zPos)
    for i in range(numberOfRandomAtomPillars - 1):
        xPos += 1 + random()*2 
        yPos += 1 + random()*2 
        xPos += 1 + random()*2
        atomPillar.extend(createAtomPillar(xPos = xPos, yPos = yPos, zPos = zPos))
    atomPillar_011 = surface(atomPillar, indices=(0, 1, 1), layers=2, periodic=True)
    atomPillar_slab = moveAndRotateAtomsAndOrthogonalize(atomPillar_011, xPos, yPos, zPos, ortho=True)
    return atomPillar_slab

def MarcelsEx(xPos = None, yPos = None, zPos = None):
    cnt1 = nanotube(10, 4, length=4)
    cnt2 = nanotube(21, 0, length=6)
    double_walled_cnt =  cnt1 + cnt2
    double_walled_cnt.rotate(-90, 'x', rotate_cell=True)
    double_walled_cnt.center(vacuum=5, axis=(0,1))
    orthogonal_atoms = moveAndRotateAtomsAndOrthogonalize(double_walled_cnt,xPos, yPos, zPos, ortho=True)
    return orthogonal_atoms

# def grapheneC(xPos = None, yPos = None, zPos = None) -> Atoms:
#     try:
#         grapheneC = read('structures/graphene.cif')
#     except FileNotFoundError:
#         grapheneC = read('FullPixelGridML/structures/graphene.cif')
#     grapheneC_101 = surface(grapheneC, indices=(1, 0, 1), layers=5, periodic=True)
#     grapheneC_slab = moveAndRotateAtomsAndOrthogonalize(grapheneC_101, xPos, yPos, zPos)
#     return grapheneC_slab

# def MoS2(xPos = None, yPos = None, zPos = None) -> Atoms:
#     try:
#         molybdenum_sulfur = read('structures/MoS2.cif')
#     except FileNotFoundError:
#         molybdenum_sulfur = read('FullPixelGridML/structures/MoS2.cif')
#     molybdenum_sulfur_011 = surface(molybdenum_sulfur, indices=(0, 1, 1), layers=2, periodic=True)
#     molybdenum_sulfur_slab = moveAndRotateAtomsAndOrthogonalize(molybdenum_sulfur_011, xPos, yPos, zPos)
#     return molybdenum_sulfur_slab

# def Si(xPos = None, yPos = None, zPos = None):
#     try:
#         silicon = read('structures/Si.cif')
#     except FileNotFoundError:
#         silicon = read('FullPixelGridML/structures/Si.cif')
#     silicon_011 = surface(silicon, indices=(0, 1, 1), layers=2, periodic=True)
#     silicon_slab = moveAndRotateAtomsAndOrthogonalize(silicon_011, xPos, yPos, zPos)
#     return silicon_slab

# def copper(xPos=None, yPos=None, zPos=None) -> Atoms:
#     try:
#         copper = read('structures/Cu.cif')
#     except FileNotFoundError:
#         copper = read('FullPixelGridML/structures/Cu.cif')
#     copper_111 = surface(copper, indices=(1, 1, 1), layers=2, periodic=True)
#     copper_slab = moveAndRotateAtomsAndOrthogonalize(copper_111, xPos, yPos, zPos)
#     return copper_slab

# def iron(xPos=None, yPos=None, zPos=None) -> Atoms:
#     try:
#         iron = read('structures/Fe.cif')
#     except FileNotFoundError:
#         iron = read('FullPixelGridML/structures/Fe.cif')
#     iron_111 = surface(iron, indices=(1, 1, 1), layers=2, periodic=True)
#     iron_slab = moveAndRotateAtomsAndOrthogonalize(iron_111, xPos, yPos, zPos)
#     return iron_slab

# def GaAs(xPos = None, yPos = None, zPos = None):
#     try:
#         gaas = read('structures/GaAs.cif')
#     except FileNotFoundError:
#         gaas = read('FullPixelGridML/structures/GaAs.cif')
#     gaas_110 = surface(gaas, indices=(1, 1, 0), layers=2, periodic=True)
#     gaas_slab = moveAndRotateAtomsAndOrthogonalize(gaas_110, xPos, yPos, zPos)
#     return gaas_slab

# def SrTiO3(xPos = None, yPos = None, zPos = None):
#     try:
#         srtio3 = read('structures/SrTiO3.cif')
#     except FileNotFoundError:
#         srtio3 = read('FullPixelGridML/structures/SrTiO3.cif')
#     srtio3_110 = surface(srtio3, indices=(1, 1, 0), layers= 5, periodic=True)
#     srtio3_slab = moveAndRotateAtomsAndOrthogonalize(srtio3_110, xPos, yPos, zPos)
#     return srtio3_slab

# def MAPbI3(xPos = None, yPos = None, zPos = None):
#     try:
#         mapi = read('structures/H6PbCI3N.cif')
#     except FileNotFoundError:
#         mapi = read('FullPixelGridML/structures/H6PbCI3N.cif')
#     mapi_110 = surface(mapi, indices=(1, 1, 0), layers=2, periodic=True)
#     mapi_slab = moveAndRotateAtomsAndOrthogonalize(mapi_110, xPos, yPos, zPos)
#     return mapi_slab

# def WSe2(xPos = None, yPos = None, zPos = None):
#     try:
#         wse2 = read('structures/WSe2.cif')
#     except FileNotFoundError:
#         wse2 = read('FullPixelGridML/structures/WSe2.cif')
#     wse2_110 = surface(wse2, indices=(1, 1, 0), layers=2, periodic=True)
#     wse2_slab = moveAndRotateAtomsAndOrthogonalize(wse2_110, xPos, yPos, zPos)
#     return wse2_slab

def StructureUnknown(**kwargs):
    raise Exception(f"Structure unknown")

def createStructure(specificStructure : str = "random", trainOrTest = None, **kwargs) -> Tuple[str, Atoms]:
    """ Creates a specified structure. If structure is unknown an Exception will be thrown. Default is "random" which randomly picks a structure


    Args:
        specificStructure (str, optional): Give structure. Defaults to "random".

        other arguments: other arguments are possible depending on the structure

    Returns:
        Atoms: Ase Atoms object of specified structure
    """
    predefinedFunctions = {
        "createAtomPillar" : createAtomPillar,
        "multiPillars" : multiPillars,
        "MarcelsEx" : MarcelsEx 
    }
    if specificStructure != "random":
        nameStruct = specificStructure
        structFinished = predefinedFunctions[nameStruct](**kwargs)
    else:
        if os.path.exists('FullPixelGridML/structures'):
            path = 'FullPixelGridML/structures'
        else:
            path = 'structures'
        cifFiles = glob.glob(os.path.join(path,"*.cif"))
        randomNumber = randint(0, len(cifFiles) - 1 + 3)
        if randomNumber >= len(cifFiles):
            nameStruct = choice(list(predefinedFunctions.keys()))
            structFinished = predefinedFunctions[nameStruct](**kwargs)
        else:
            cifFile = cifFiles[randomNumber]
            struct = read(cifFile)
            nameStruct = cifFile.split("\\")[-1].split(".")[0]
            #create a tuple with 3 random numbers, either one or zero
            random_numbers = (randint(0, 1), randint(0, 1), randint(0, 1))
            if random_numbers == (0,0,0):
                random_numbers = (1,1,1)
            surfaceStruct = surface(struct, indices=random_numbers, layers=3, periodic=True)
            structFinished = moveAndRotateAtomsAndOrthogonalize(surfaceStruct)
    return nameStruct, structFinished

def generateDiffractionArray(trainOrTest = None, conv_angle = 33, energy = 60e3, structure = "random", pbar = False, start = (0,0), end = (-1,-1)):

    nameStruct, atomStruct = createStructure(specificStructure= structure, trainOrTest = trainOrTest)
    try:
        potential_thick = Potential(
            atomStruct,
            sampling=0.05,
            device=device
        )
    except Exception as e:
        print(nameStruct, atomStruct)
        raise(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = Probe(semiangle_cutoff=conv_angle, energy=energy, device=device)
        probe.match_grid(potential_thick)

        pixelated_detector = PixelatedDetector(max_angle=100,resample = "uniform")
        gridSampling = (0.2,0.2)
        if end == (-1,-1):
            end = potential_thick.extent
        gridscan = GridScan(
            start = start, end = end, sampling=gridSampling
        )
        measurement_thick = probe.scan(gridscan, pixelated_detector, potential_thick, pbar = pbar)

        # plt.imsave("difPattern.png", measurement_thick.array[0,0])
        #TODO: add noise
        #We dont give angle, conv_angle, energy, real pixelsize to ai because its the same for all training data. Can be done in future

        return nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick

@njit
def threeClosestAtoms(atomPositions:np.ndarray, atomicNumbers:np.ndarray, xPos:float, yPos:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Distances = (atomPositions - np.expand_dims(np.array([xPos, yPos, 0]),0))[:,0:2]
    Distances = Distances[:,0] + Distances[:,1] * 1j
    DistanceSortedIndices = np.absolute(Distances).argsort() 

    # DistancesSqu = np.array(Distances)**2
    # DistanceSortedIndices = DistancesSqu.sum(axis=1).argsort()

    while len(DistanceSortedIndices) < 3: #if less than three atoms, just append the closest one again
        DistanceSortedIndices = np.concatenate((DistanceSortedIndices,DistanceSortedIndices))
    
    atomNumbers = atomicNumbers[DistanceSortedIndices[:3]]
    xPositions, yPositions = atomPositions[DistanceSortedIndices[:3]].transpose()[:2]
    return atomNumbers, xPositions, yPositions

@njit
def closestAtom(atomPositions:np.ndarray, atomicNumbers:np.ndarray, xPos:float, yPos:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Distances = (atomPositions - np.expand_dims(np.array([xPos, yPos, 0]),0))[:,0:2]
    Distances = Distances[:,0] + Distances[:,1] * 1j
    DistanceSortedIndices = np.absolute(Distances).argsort() 
    
    atomNumber = atomicNumbers[DistanceSortedIndices[0]]
    xPosition, yPosition = atomPositions[DistanceSortedIndices[0]][:2]
    return atomNumber, xPosition, yPosition

@njit
def findAtomsInTile(xPosTile:float, yPosTile:float, xRealLength:float, yRealLength:float, atomPositions:np.ndarray):
    atomPositionsCopy = np.copy(atomPositions)
    xAtomPositions, yAtomPositions, _ = atomPositionsCopy.transpose()
    atomPosInside = atomPositions[(xPosTile <= xAtomPositions) * (xAtomPositions < xPosTile + xRealLength) * (yPosTile <= yAtomPositions) * (yAtomPositions < yPosTile + yRealLength)]
    xPositions, yPositions, _ = atomPosInside.transpose()
    return xPositions, yPositions

def createAllXYCoordinates(yMaxCoord, xMaxCoord):
    return [(x,y) for y in np.arange(yMaxCoord+1) for x in np.arange(xMaxCoord+1)]


def generateGroundThruthRelative(rows, atomStruct, datasetStructID, xRealLength, yRealLength, xSteps, ySteps, xPosTile, yPosTile):
    #generates row in labels.csv with relative distances of three closest atoms to the center of the diffraction pattern. Also saves the element predictions.
    atomNumber, xPositionsAtom, yPositionsAtom = closestAtom(atomStruct.get_positions(), atomStruct.get_atomic_numbers(), xPosTile + xRealLength/2, yPosTile + yRealLength/2)
                
    xAtomRel = xPositionsAtom - xPosTile
    yAtomRel = yPositionsAtom - yPosTile
    rows.append([f"{datasetStructID}[{xSteps}][{ySteps}]"] + [str(difParams) for difParams in [atomNumber, xAtomRel, yAtomRel]])
    return rows

def generateGroundThruthPixel(rows, XDIMTILES, YDIMTILES, atomStruct, datasetStructID, xRealLength, yRealLength, xCoord, yCoord, xPosTile, yPosTile, maxPooling = 1):
    #Generates row in labels.csv with XDIMTILES*YDIMTILES pixels. Each pixel is one if an atom is in the pixel and zero if not.
    pixelGrid = np.zeros((XDIMTILES, YDIMTILES), dtype = int)
    xPositions, yPositions = findAtomsInTile(xPosTile, yPosTile, xRealLength, yRealLength, atomStruct.get_positions())
    #integer division is good enough. Othewise we would have to use the real pixel size in findAtomsInTile to get the correct boundaries
    xPositions = (xPositions - xPosTile) / (xRealLength/XDIMTILES)
    yPositions = (yPositions - yPosTile) / (yRealLength/YDIMTILES)
    xPositions = xPositions.astype(int)
    yPositions = yPositions.astype(int)
    #print the maxima of the x and y positions
    for x,y in zip(xPositions, yPositions):
        pixelGrid[x, y] = 1
    if maxPooling > 1:
        pixelGrid = block_reduce(pixelGrid, maxPooling, np.max)
    rows.append([f"{datasetStructID}[{xCoord}][{yCoord}]"] + [str(pixel) for pixel in pixelGrid.flatten()])
    return rows

def saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, numberOfPatterns, timeStamp, BFDdiameter, processID = 99999, silence = False, maxPooling = 1, structure = "random"):
    rows = []
    xStepSize = (XDIMTILES-1)//3
    yStepSize = (YDIMTILES-1)//3
    dim = 50
    # allTiles = XDIMTILES * YDIMTILES
    # fractionOfNonZeroIndices = {}
    # with open(os.path.join(f'measurements_{trainOrTest}',f'fractionOfNonZeroIndices_{processID}_{timeStamp}.csv'), 'w', newline='') as csvfile:
    #     Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     Writer.writerow(["datasetStructID", "Fraction of non-zero indices"])
    with h5py.File(os.path.join(f"measurements_{trainOrTest}",f"{processID}_{timeStamp}.hdf5"), 'w') as file:
        for cnt, (nameStruct, gridSampling, atomStruct, measurement_thick, _) in enumerate((generateDiffractionArray(trainOrTest = trainOrTest, structure=structure) for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))):
            datasetStructID = f"{cnt}{processID}{timeStamp}" 
        
            xRealLength = XDIMTILES * gridSampling[0]
            yRealLength = YDIMTILES * gridSampling[1]

            xMaxCNT, yMaxCNT = np.shape(measurement_thick.array)[:2] # type: ignore
            xMaxStartCoord = (xMaxCNT-XDIMTILES)//xStepSize
            yMaxStartCoord = (yMaxCNT-YDIMTILES)//yStepSize

            if xMaxStartCoord < 0 or yMaxStartCoord < 0:
                print(f"Structure too small, skipped: xMaxStartCoord : {xMaxStartCoord}, yMaxStartCoord : {yMaxStartCoord}, struct {nameStruct}, np.shape(measurement_thick.array)[:2] : {np.shape(measurement_thick.array)[:2]}") # type: ignore
                continue

            for xSteps, ySteps in tqdm(createAllXYCoordinates(yMaxStartCoord,xMaxStartCoord), leave=False,desc = f"Going through diffraction Pattern in {XDIMTILES}x{YDIMTILES} tiles {processID}.", total= len(measurement_thick.array), disable=silence): # type: ignore
                xCNT = xStepSize * xSteps
                yCNT = yStepSize * ySteps

                difPatternsOnePosition = measurement_thick.array[xCNT:xCNT + XDIMTILES:3, yCNT :yCNT + YDIMTILES:3].copy() # type: ignore
                difPatternsOnePosition = np.reshape(difPatternsOnePosition, (-1,difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))
                # randomIndicesToTurnToZeros = np.random.choice(difPatternsOnePosition.shape[0], randint(allTiles - int(0.4*allTiles),allTiles - int(0.4*allTiles)), replace = False)
                # difPatternsOnePosition[randomIndicesToTurnToZeros] = np.zeros((difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))            

                xPosTile = xCNT * gridSampling[0]
                yPosTile = yCNT * gridSampling[1]

                rows = generateGroundThruthRelative(rows, atomStruct, datasetStructID, xRealLength, yRealLength, xSteps, ySteps, xPosTile, yPosTile)
                #fractionOfNonZeroIndices[f"{datasetStructID}[{xSteps}][{ySteps}]"] = (len(difPatternsOnePosition) - len(randomIndicesToTurnToZeros))/len(difPatternsOnePosition)
                
                # rows = generateGroundThruthPixel(rows, XDIMTILES, YDIMTILES, atomStruct, datasetStructID, xRealLength, yRealLength, xSteps, ySteps, xPosTile, yPosTile, maxPooling = maxPooling)
                

                difPatternsOnePositionResized = []
                for cnt, difPattern in enumerate(difPatternsOnePosition): 
                   difPatternsOnePositionResized.append(cv2.resize(np.array(difPattern), dsize=(dim, dim), interpolation=cv2.INTER_LINEAR))  # type: ignore
                
                difPatternsOnePositionResized = np.array(difPatternsOnePositionResized)
                # removing everything outside the bright field disk
                indicesInBFD = slice(max((dim - BFDdiameter)//2-1,0),min((dim + BFDdiameter)//2+1, dim ))
                difPatternsOnePositionResized = difPatternsOnePositionResized[:,indicesInBFD, indicesInBFD] 
                # plt.imsave(os.path.join(f"measurements_{trainOrTest}",f"{datasetStructID}.png"), difPatternsOnePositionResized[0])

                file.create_dataset(f"{datasetStructID}[{xSteps}][{ySteps}]", data = difPatternsOnePositionResized, compression="lzf", chunks = (1, difPatternsOnePositionResized.shape[-2], difPatternsOnePositionResized.shape[-1]), shuffle = True)
    # with open(os.path.join(f'measurements_{trainOrTest}',f'fractionOfNonZeroIndices_{processID}_{timeStamp}.csv'), 'w+', newline='') as csvfile:
    #     Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for key, value in fractionOfNonZeroIndices.items():
    #         Writer.writerow([key, value])
    return rows            

def createTopLineRelative(csvFilePath):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #Writer.writerow(["fileName", "element1", "element2", "element3", "xAtomRel1", "xAtomRel2", "xAtomRel3", "yAtomRel1", "yAtomRel2", "yAtomRel3"])
        Writer.writerow(["fileName", "element", "xAtomRel", "yAtomRel"])
        Writer = None

def createTopLinePixels(csvFilePath, XDIMTILES, YDIMTILES, maxPooling = 1):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName"]+ [f"pixelx{x}y{y}" for y in np.arange(XDIMTILES//maxPooling) for x in np.arange(YDIMTILES//maxPooling)])
        Writer = None    

def writeAllRows(rows, trainOrTest, XDIMTILES, YDIMTILES, processID = "", createTopRow = None, timeStamp = 0, maxPooling = 1):
    """
    Writes the given rows to a CSV file.

    Args:
        rows (list): A list of rows to be written to the CSV file.
        trainOrTest (str): Indicates whether the data is for training or testing purposes.
        processID (str, optional): An optional identifier for the process. Defaults to an empty string.
        createTopRow (bool, optional): Specifies whether to create a top row in the CSV file. If None,
            it checks if the file already exists and creates the top row if it doesn't. Defaults to None.

    Returns:
        None
    """
    csvFilePath = os.path.join(f'measurements_{trainOrTest}',f'labels_{processID}_{timeStamp}.csv')
    if createTopRow is None: createTopRow = not os.path.exists(csvFilePath)
    if createTopRow: createTopLineRelative(csvFilePath)
    # if createTopRow: createTopLinePixels(csvFilePath, XDIMTILES, YDIMTILES, maxPooling = maxPooling)
    with open(csvFilePath, 'a', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            Writer.writerow(row)



if __name__ == "__main__":
    print("Running")
    import argparse
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("-id", "--id", type=str, required=False, default= "0",help="version number")
    ap.add_argument("-it", "--iterations", type=int, required=False, default= 1,help="number of iterations")
    ap.add_argument("-t", "--trainOrTest", type=str, required = False, default="traintest", help="specify train or test if you want to limit to just one")
    ap.add_argument("-s", "--structure", type=str, required = False, default="random", help="Specify if a specific structure should be used. Otherwise random will be chosen.")
    args = vars(ap.parse_args())


    XDIMTILES = 11
    YDIMTILES = 11
    maxPooling = 3
    BFDdiameter = 18 #chosen on the upper end of the BFD diameters (like +4) to have a good margin
    # assert(XDIMTILES % maxPooling == 0)
    testDivider = {"train":1, "test":0.25}
    for i in tqdm(range(max(args["iterations"],1)), disable=True):
        for trainOrTest in ["train", "test"]:
            if trainOrTest not in args["trainOrTest"]:
                continue
            print(f"PID {os.getpid()} on step {i+1} of {trainOrTest}-data at {datetime.datetime.now()}")
            with(open(f"progress_{args['id']}.txt", "w")) as file:
                file.write(f"PID {os.getpid()} on step {i+1} of {trainOrTest}-data at {datetime.datetime.now()}\n")
            timeStamp = int(str(time()).replace('.', ''))
            rows = saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, BFDdiameter, processID=args["id"], silence=True, maxPooling = maxPooling, structure = args["structure"])
            writeAllRows(rows=rows, trainOrTest=trainOrTest, XDIMTILES=XDIMTILES, YDIMTILES=YDIMTILES, processID=args["id"], timeStamp = timeStamp, maxPooling=maxPooling)
  
    print(f"PID {os.getpid()} done.")

        
# measurement_noise = poisson_noise(measurement_thick, 1e6)
# for 
#     fileName = "measurements\\{element}_{i}_{xAtom}_{xAtomShift}_{yAtom}_{yAtomShift}"
#     fileName.format(element = element, i = i, xAtom = xAtom, xAtomShift = xAtomShift, yAtom = yAtom, yAtomShift = yAtomShift)
#     for cntRow, row in enumerate(measurement_noise.array):
#         for cntClmn, difPat in enumerate(row):
#             xPos = cntRow * 0.2

#     for difPattern in measurement_noise.array:
#         difPattern.save_as_image(fileName)
# slice_thicknesses = atomPillar.cell.lengths()[-1] / 1.*zAtoms
# multislice_reconstruction_ptycho_operator = MultislicePtychographicOperator(
#     measurement_thick,
#     semiangle_cutoff=24,
#     energy=200e3,
#     num_slices=zAtoms,
#     device="gpu",
#     slice_thicknesses=slice_thicknesses,
#     parameters={"object_px_padding": (0, 0)},
# ).preprocess()

# (
#     mspie_objects,
#     mspie_probes,
#     mspie_positions,
#     mspie_sse,
# ) = multislice_reconstruction_ptycho_operator.reconstruct(
#     max_iterations=5,
#     verbose=True,
#     random_seed=1,
#     return_iterations=True,
#     parameters={
#         "pure_phase_object_update_steps": multislice_reconstruction_ptycho_operator._num_diffraction_patterns
#     },
# )

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# mspie_objects[-1].angle().sum(0).interpolate(potential_thick.sampling).show(
#     cmap="magma", ax=ax1, title=f"SSE = {float(mspie_sse[-1]):.3e}"
# )
# mspie_probes[-1][0].intensity().interpolate(potential_thick.sampling).show(ax=ax2)

# fig.tight_layout()

# plt.show()