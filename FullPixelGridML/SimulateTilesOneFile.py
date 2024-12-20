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
from random import random, randint, choice, uniform, randrange
from abtem.reconstruct import MultislicePtychographicOperator, RegularizedPtychographicOperator
from abtem import Potential, FrozenPhonons, Probe, CTF
from abtem.detect import AnnularDetector, PixelatedDetector
from abtem.scan import GridScan
from abtem.noise import poisson_noise
from abtem.structures import orthogonalize_cell
import abtem
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
xlen = ylen = 25
pixelOutput = False

def calc_diameter_bfd(image):
    brightFieldDisk = np.zeros_like(image)
    brightFieldDisk[image > np.max(image)*0.1 + np.min(image)*0.9] = 1
    bfdArea = np.sum(brightFieldDisk)
    diameterBFD = np.sqrt(bfdArea/np.pi) * 2
    return diameterBFD

def moveAndRotateAtomsAndOrthogonalizeAndRepeat(atoms : Atoms, xPos = None, yPos = None, zPos = None, ortho = True) -> Atoms:
    xPos = random()*xlen/3 + xlen/3 if xPos is None else xPos
    yPos = random()*ylen/3 + ylen/3 if yPos is None else yPos
    zPos = 0 if zPos is None else zPos
    atoms.positions += np.array([xPos, yPos, zPos])[None,:]    
    atoms.rotate("x", randint(0,360)) 
    atoms.rotate("y", randint(0,360))  
    atoms.rotate("z", randint(0,360))
    if ortho: atoms = orthogonalize_cell(atoms, max_repetitions=10)
    xLength, yLength, zLength = atoms.cell.lengths()
    atoms_slab = atoms.repeat((int(max(xlen/xLength, 1)), int(max(ylen/yLength,1)), 1))
    return atoms_slab # type: ignore

def createAtomPillar(xPos = None, yPos = None, zPos = None, zAtoms = randint(1,10), xAtomShift = 0, yAtomShift = 0, element = None) -> Atoms:
    element = randint(1, 100) if element is None else element
    # maxShiftFactorPerLayer = 0.01
    #(random()  - 1/2) * maxShiftFactorPerLayer * xlen  #slant of atom pillar
    #(random() -  1/2) * maxShiftFactorPerLayer * ylen #slant of atom pillar
    # xAtomPillarAngle = np.arcsin(xAtomShift/1)
    # yAtomPillarAngle = np.arcsin(yAtomShift/1)
    positions = [np.array([xPos or 0, yPos or 0, zPos or 0])]
    for atomBefore in range(zAtoms-1):
        positions += [positions[atomBefore] + np.array([xAtomShift, yAtomShift, 1])]
    atomPillar = Atoms(numbers = [element] * zAtoms , positions=positions, cell = [xlen, ylen, zAtoms,90,90,90])
    #atomPillar_101 = surface(atomPillar, indices=(1, 0, 1), layers=2, periodic=True)
    # atomPillar = moveAndRotateAtomsAndOrthogonalizeAndRepeat(atomPillar, xPos, yPos, zPos, ortho=True)
    return atomPillar

#BROKEN #TODO use poisson disk
def multiPillars(xAtomShift = 0, yAtomShift = 0, element = None, numberOfRandomAtomPillars = None) -> Atoms:
    numberOfRandomAtomPillars = numberOfRandomAtomPillars or randint(25,50)
    xPos = random()*ylen 
    yPos = random()*xlen
    zPos = 0
    atomPillar = createAtomPillar(xPos = xPos, yPos = yPos, zPos = zPos)
    for _ in range(numberOfRandomAtomPillars - 1):
        xPos = random()*ylen 
        yPos = random()*xlen
        atomPillar.extend(createAtomPillar(xPos = xPos, yPos = yPos, zPos = zPos))
    #atomPillar_011 = surface(atomPillar, indices=(0, 1, 1), layers=2, periodic=True)
    atomPillar_slab = moveAndRotateAtomsAndOrthogonalizeAndRepeat(atomPillar, xPos, yPos, zPos, ortho=True)
    return atomPillar_slab

def MarcelsEx(xPos = None, yPos = None, zPos = None):
    cnt1 = nanotube(10, 4, length=4)
    cnt2 = nanotube(21, 0, length=6)
    double_walled_cnt =  cnt1 + cnt2
    double_walled_cnt.rotate(-90, 'x', rotate_cell=True)
    double_walled_cnt.center(vacuum=5, axis=(0,1))
    orthogonal_atoms = moveAndRotateAtomsAndOrthogonalizeAndRepeat(double_walled_cnt,xPos, yPos, zPos, ortho=True)
    return orthogonal_atoms

def grapheneC(xPos = None, yPos = None, zPos = None) -> Atoms:
    grapheneC = graphene()
    grapheneC_slab = moveAndRotateAtomsAndOrthogonalizeAndRepeat(grapheneC, xPos, yPos, zPos)
    return grapheneC_slab

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
    simple = False
    if simple:
        nameStruct = "createAtomPillar"
        structFinished = createAtomPillar(xPos=5+random()*3, yPos=5+random()*3, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100))
        for _ in range(9):
            structFinished.extend(createAtomPillar(xPos=5+random()*3, yPos=5+random()*3, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100)))
        # structFinished = createAtomPillar(xPos=5.1, yPos=5.1, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100))
        # for _ in range(9):
        #     structFinished.extend(createAtomPillar(xPos=5.3+_*0.2, yPos=5.3+_*0.2, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100)))
        
        structFinished = orthogonalize_cell(structFinished, max_repetitions=10)
        return nameStruct, structFinished
    predefinedFunctions = {
        "createAtomPillar" : createAtomPillar,
        "multiPillars" : multiPillars,
        "MarcelsEx" : MarcelsEx,
        "grapheneC" : grapheneC,
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
            struct  = read(cifFile)
            nameStruct = cifFile.split("\\")[-1].split(".")[0]
            #create a tuple with 3 random numbers, either one or zero
            # random_numbers = (randint(0, 1), randint(0, 1), randint(0, 1))
            # if random_numbers == (0,0,0):
            #     random_numbers = (1,1,1)
            #surfaceStruct = surface(struct, indices=random_numbers, layers=3, periodic=True)
            structFinished = moveAndRotateAtomsAndOrthogonalizeAndRepeat(struct)
    return nameStruct, structFinished

def generateDiffractionArray(trainOrTest = None, conv_angle = 33, energy = 60e3, structure = "random", pbar = False, start = (5,5), end = (20,20)) -> Tuple[str, Tuple[float, float], Atoms, Measurement, Potential]:

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

# @njit
# def threeClosestAtoms(atomPositions:np.ndarray, atomicNumbers:np.ndarray, xPos:float, yPos:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     Distances = (atomPositions - np.expand_dims(np.array([xPos, yPos, 0]),0))[:,0:2]
#     Distances = Distances[:,0] + Distances[:,1] * 1j
#     DistanceSortedIndices = np.absolute(Distances).argsort() 

#     # DistancesSqu = np.array(Distances)**2
#     # DistanceSortedIndices = DistancesSqu.sum(axis=1).argsort()

#     while len(DistanceSortedIndices) < 3: #if less than three atoms, just append the closest one again
#         DistanceSortedIndices = np.concatenate((DistanceSortedIndices,DistanceSortedIndices))
    
#     atomNumbers = atomicNumbers[DistanceSortedIndices[:3]]
#     xPositions, yPositions = atomPositions[DistanceSortedIndices[:3]].transpose()[:2]
#     return atomNumbers, xPositions, yPositions

@njit
def findAtomsInTile(xPosTile:float, yPosTile:float, xRealLength:float, yRealLength:float, atomPositions:np.ndarray):
    atomPositionsCopy = np.copy(atomPositions)
    xAtomPositions, yAtomPositions, _ = atomPositionsCopy.transpose()
    atomPosInside = atomPositions[(xPosTile <= xAtomPositions) * (xAtomPositions < xPosTile + xRealLength) * (yPosTile <= yAtomPositions) * (yAtomPositions < yPosTile + yRealLength)]
    xPositions, yPositions, _ = atomPosInside.transpose()
    return xPositions, yPositions

def createAllXYCoordinates(yMaxCoord, xMaxCoord):
    return [(x,y) for y in np.arange(yMaxCoord+1) for x in np.arange(xMaxCoord+1)]

#@njit
def closestAtoms(atomPositions:np.ndarray, atomicNumbers:np.ndarray, xPosTileCenter:float, 
                yPosTileCenter:float)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the closest atoms to a given tile center position.

    Parameters:
    - atomPositions (np.ndarray): Array of atom positions.
    - atomicNumbers (np.ndarray): Array of atomic numbers.
    - xPosTileCenter (float): X-coordinate of the tile center.
    - yPosTileCenter (float): Y-coordinate of the tile center.

    Returns:
    - atomNumbersSorted (np.ndarray): Array of atomic numbers sorted by distance.
    - xRelativDistancesSorted (np.ndarray): Array of x-coordinate relative distances sorted by distance.
    - yRelativDistancesSorted (np.ndarray): Array of y-coordinate relative distances sorted by distance.
    """
    Distances = (atomPositions - np.expand_dims(np.array([xPosTileCenter, yPosTileCenter, 0]),0))[:,0:2]

    Distances = Distances[:,0] + Distances[:,1] * 1j
    DistanceSortedIndices = np.absolute(Distances).argsort() 
    # plt.scatter(x=xPosTileCenter, y=yPosTileCenter, c='r', label='current Position')  # use this to plot a single point
    # plt.scatter(x=atomPositions[:,0], y=atomPositions[:,1], c='black', label='atoms')
    # plt.scatter(x=atomPositions[DistanceSortedIndices[0]][0], y=atomPositions[DistanceSortedIndices[0]][1], c='g', label='closest Atom')
    # plt.legend()
    # plt.savefig("closestAtoms.png")
    
    
    atomNumbersSorted = atomicNumbers[DistanceSortedIndices]
    #xPosition, yPosition = atomPositions[DistanceSortedIndices[0]][:2] 
    xRelativDistancesSorted, yRelativDistancesSorted = Distances[DistanceSortedIndices].real, Distances[DistanceSortedIndices].imag
    # print(xRelativDistancesSorted[0], yRelativDistancesSorted[0])
    # print("current Position: ", xPosTileCenter, yPosTileCenter)
    # exit()
    return atomNumbersSorted, xRelativDistancesSorted, yRelativDistancesSorted

def generateGroundThruthRelative(rows, atomStruct, datasetStructID, xRealLength, yRealLength, xSteps, ySteps, xPosTile, yPosTile, start = (5,5)):
    #generates row in labels.csv with relative distances of three closest atoms to the center of the diffraction pattern. Also saves the element predictions.
    atomNumbersSorted, xRelativDistancesSorted, yRelativDistancesSorted = closestAtoms(atomStruct.get_positions(), atomStruct.get_atomic_numbers(), xPosTile + xRealLength/2 + start[0], yPosTile + yRealLength/2 + start[1])
    
    # print(f"atomStruct.get_positions(): {atomStruct.get_positions()}")
    # print(f"atomStruct.get_cell(): {atomStruct.get_cell()}")
    # print(f"atomNumber: {atomNumber}, xPositionsAtom: {xPositionsAtom}, yPositionsAtom: {yPositionsAtom}, xPosTile: {xPosTile}, yPosTile: {yPosTile}, xRealLength: {xRealLength}, yRealLength: {yRealLength}")
    # abtem.show_atoms(
    # atomStruct
    # )
    # plt.savefig("atomStructTest.png")
    # exit()
    if len(atomNumbersSorted) >= 4:
        rows.append([f"{datasetStructID}[{xSteps}][{ySteps}]"] + [str(element) for element in atomNumbersSorted[:4]] + [str(xDistance) for xDistance in xRelativDistancesSorted[:4]] + [str(yDistance) for yDistance in yRelativDistancesSorted[:4]])
    return rows

# def generateGroundThruthPixel(rows, XDIMTILES, YDIMTILES, atomStruct, datasetStructID, xRealLength, yRealLength, xCoord, yCoord, xPosTile, yPosTile, maxPooling = 1, start = (5,5)):
#     #Generates row in labels.csv with XDIMTILES*YDIMTILES pixels. Each pixel is one if an atom is in the pixel and zero if not.
#     pixelGrid = np.zeros((XDIMTILES, YDIMTILES), dtype = int)
#     xPositions, yPositions = findAtomsInTile(xPosTile + xRealLength/2 - start[0], yPosTile + yRealLength/2 + start[1], xRealLength, yRealLength, atomStruct.get_positions())
#     #integer division is good enough. Othewise we would have to use the real pixel size in findAtomsInTile to get the correct boundaries
#     xPositions = (xPositions - xPosTile) / (xRealLength/XDIMTILES)
#     yPositions = (yPositions - yPosTile) / (yRealLength/YDIMTILES)
#     xPositions = xPositions.astype(int)
#     yPositions = yPositions.astype(int)
#     #print the maxima of the x and y positions
#     for x,y in zip(xPositions, yPositions):
#         pixelGrid[x, y] = 1
#     if maxPooling > 1:
#         pixelGrid = block_reduce(pixelGrid, maxPooling, np.max)
#     rows.append([f"{datasetStructID}[{xCoord}][{yCoord}]"] + [str(pixel) for pixel in pixelGrid.flatten()])
#     return rows



def generateAtomGrid(datasetStructID, rows, atomStruct, start, end, silence):
    allPositions = atomStruct.get_positions()
    allPositionsShifted = allPositions - np.array([start[0], start[1], 0])
    numberOfPositionsInOneAngstrom = 5
    xMaxCoord, yMaxCoord = (end[0] - start[0])* numberOfPositionsInOneAngstrom, (end[1] - start[1]) * numberOfPositionsInOneAngstrom
    gridOfCoords = np.array([[x/numberOfPositionsInOneAngstrom+y/numberOfPositionsInOneAngstrom * 1j for y in np.arange(yMaxCoord)] for x in np.arange(xMaxCoord)])
    # print(gridOfCoords)
    # plt.scatter(x=allPositionsShifted[:,0][(allPositionsShifted[:,0]>0) *(allPositionsShifted[:,1]>0)], y=allPositionsShifted[:,1][(allPositionsShifted[:,0]>0) * (allPositionsShifted[:,1]>0)], c='black', label='atoms')

    # plt.savefig("atomStructTest.png")
    atomGridInterpolated = np.zeros((xMaxCoord, yMaxCoord))
    silence = False

    for (x, y), atomNo in tqdm(zip(allPositionsShifted[:,0:2], atomStruct.get_atomic_numbers()), leave=False, desc = f"Going through diffraction Pattern in atoms.", total= len(allPositionsShifted), disable=silence):
        x = x 
        y = y * 1j
        distance  = gridOfCoords - x - y
        distance = np.absolute(distance) + 1
        OneOverdistanceSquTimesAtomNo = 1/distance**2 * atomNo
        atomGridInterpolated += OneOverdistanceSquTimesAtomNo
    rows.append([f"{datasetStructID}"]+list(atomGridInterpolated.flatten()))
    return rows
    
def generateAtomGridNoInterp(datasetStructID, rows, atomStruct, start, end, silence, maxPooling = 1):
    allPositions = atomStruct.get_positions()
    allPositionsShifted = allPositions - np.array([start[0], start[1], 0])
    numberOfPositionsInOneAngstrom = 5
    xMaxCoord, yMaxCoord = int((end[0] - start[0])* numberOfPositionsInOneAngstrom//maxPooling), int((end[1] - start[1]) * numberOfPositionsInOneAngstrom//maxPooling)

    atomGrid= np.zeros((xMaxCoord, yMaxCoord))
    silence = False

    for (x, y), atomNo in tqdm(zip(allPositionsShifted[:,0:2], atomStruct.get_atomic_numbers()), leave=False, desc = f"Going through diffraction Pattern in atoms.", total= len(allPositionsShifted), disable=silence):
        xRound = np.round(x*numberOfPositionsInOneAngstrom/maxPooling)
        yRound = np.round(y*numberOfPositionsInOneAngstrom/maxPooling)
        if xRound < 0 or yRound < 0: continue
        if xRound >= xMaxCoord or yRound >= yMaxCoord: continue
        atomGrid[int(xRound), int(yRound)] += atomNo
    np.clip(atomGrid, 0, 1, out=atomGrid) #clip to 1 for classifier
    rows.append([f"{datasetStructID}"]+list(atomGrid.flatten()))
    return rows

#a function that appends the xy positions of the atoms to the rows
def generateXYE(datasetStructID, rows, atomStruct, start, end, silence):
    allPositions = atomStruct.get_positions()
    allPositionsShifted = allPositions - np.array([start[0], start[1], 0])
    numberOfPositionsInOneAngstrom = 5
    silence = False
    xMaxCoord, yMaxCoord = int((end[0] - start[0])* numberOfPositionsInOneAngstrom), int((end[1] - start[1]) * numberOfPositionsInOneAngstrom)
    xyes = []
    cnt = 0
    for (x, y), atomNo in tqdm(zip(allPositionsShifted[:,0:2], atomStruct.get_atomic_numbers()), leave=False, desc = f"Going through diffraction Pattern in atoms.", total= len(allPositionsShifted), disable=silence):
        xCoordinate = x*numberOfPositionsInOneAngstrom
        yCoordinate = y*numberOfPositionsInOneAngstrom
        
        if xCoordinate < 0 or yCoordinate < 0: continue
        if xCoordinate > xMaxCoord or yCoordinate > yMaxCoord: continue
        xyes.append([xCoordinate,yCoordinate,atomNo])
        cnt += 1
    xyes = np.array(xyes)
    xyes = xyes[xyes[:,0].argsort()] #sort by x coordinate

    rows.append([f"{datasetStructID}"]+list(xyes.flatten().astype(str)))    
    return rows

def saveAllPosDifPatterns(trainOrTest, numberOfPatterns, timeStamp, BFDdiameter,maxPooling = 1, processID = 99999, silence = False, structure = "random", fileWrite = True, difArrays = None,     start = (5,5), end = (8,8)):
    rows = []
    # rowsPixel = []
    dim = 50


    if fileWrite: file = h5py.File(os.path.join(f"measurements_{trainOrTest}",f"{processID}_{timeStamp}.hdf5"), 'w')
    else: dataArray = []
    difArrays = difArrays or (generateDiffractionArray(trainOrTest = trainOrTest, structure=structure, start=start, end=end) for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))
    for cnt, (nameStruct, gridSampling, atomStruct, measurement_thick, _) in enumerate(difArrays):
        datasetStructID = f"{cnt}{processID}{timeStamp}"         

        difPatternsOnePosition = measurement_thick.array.copy() # type: ignore
        difPatternsOnePosition = np.reshape(difPatternsOnePosition, (-1,difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))
        if pixelOutput:
            rows = generateAtomGridNoInterp(datasetStructID, rows, atomStruct, start, end, silence, maxPooling=maxPooling)
        else:
            rows = generateXYE(datasetStructID, rows, atomStruct, start, end, silence)
        difPatternsOnePositionResized = []
        for cnt, difPattern in enumerate(difPatternsOnePosition): 
            difPatternsOnePositionResized.append(cv2.resize(np.array(difPattern), dsize=(dim, dim), interpolation=cv2.INTER_LINEAR))  # type: ignore
        
        difPatternsOnePositionResized = np.array(difPatternsOnePositionResized)
        # removing everything outside the bright field disk
        indicesInBFD = slice(max((dim - BFDdiameter)//2-1,0),min((dim + BFDdiameter)//2+1, dim ))
        difPatternsOnePositionResized = difPatternsOnePositionResized[:,indicesInBFD, indicesInBFD] 
        # plt.imsave(os.path.join(f"measurements_{trainOrTest}",f"{datasetStructID}.png"), difPatternsOnePositionResized[0])
        
        if fileWrite: file.create_dataset(f"{datasetStructID}", data = difPatternsOnePositionResized, compression="lzf", chunks = (1, difPatternsOnePositionResized.shape[-2], difPatternsOnePositionResized.shape[-1]), shuffle = True)
        else: dataArray.append(difPatternsOnePositionResized)
    if fileWrite: 
        file.close()
        return rows
    else:
        return rows, np.array(dataArray) 

def saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, numberOfPatterns, timeStamp, BFDdiameter, processID = 99999, silence = False, maxPooling = 1, structure = "random", fileWrite = True, difArrays = None, start = (5,5), end = (20,20)):
    rowsRelative = []
    # rowsPixel = []
    stepsPerTile = 3
    xStepSize = (XDIMTILES-1)//stepsPerTile
    yStepSize = (YDIMTILES-1)//stepsPerTile
    dim = 50
    
    if fileWrite: file = h5py.File(os.path.join(f"measurements_{trainOrTest}",f"{processID}_{timeStamp}.hdf5"), 'w')
    else: dataArray = []
    difArrays = difArrays or (generateDiffractionArray(trainOrTest = trainOrTest, structure=structure, start=start, end=end) for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))
    for cnt, (nameStruct, gridSampling, atomStruct, measurement_thick, _) in enumerate(difArrays):
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

            rowsRelative = generateGroundThruthRelative(rowsRelative, atomStruct, datasetStructID, xRealLength, yRealLength, xSteps, ySteps, xPosTile, yPosTile, start)            
            # rowsPixel = generateGroundThruthPixel(rowsPixel, XDIMTILES, YDIMTILES, atomStruct, datasetStructID, xRealLength, yRealLength, xSteps, ySteps, xPosTile, yPosTile, maxPooling = maxPooling)
            

            difPatternsOnePositionResized = []
            for cnt, difPattern in enumerate(difPatternsOnePosition): 
                difPatternsOnePositionResized.append(cv2.resize(np.array(difPattern), dsize=(dim, dim), interpolation=cv2.INTER_LINEAR))  # type: ignore
            
            difPatternsOnePositionResized = np.array(difPatternsOnePositionResized)
            # removing everything outside the bright field disk
            indicesInBFD = slice(max((dim - BFDdiameter)//2-1,0),min((dim + BFDdiameter)//2+1, dim ))
            difPatternsOnePositionResized = difPatternsOnePositionResized[:,indicesInBFD, indicesInBFD] 
            # plt.imsave(os.path.join(f"measurements_{trainOrTest}",f"{datasetStructID}.png"), difPatternsOnePositionResized[0])
            
            if fileWrite: file.create_dataset(f"{datasetStructID}[{xSteps}][{ySteps}]", data = difPatternsOnePositionResized, compression="lzf", chunks = (1, difPatternsOnePositionResized.shape[-2], difPatternsOnePositionResized.shape[-1]), shuffle = True)
            else: dataArray.append(difPatternsOnePositionResized)
    if fileWrite: 
        file.close()
        return rowsRelative
    else:
        return rowsRelative, np.array(dataArray) 

def createTopLineCoords(csvFilePath):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        row = ["fileName"]
        for i in range(10):
            row = row + [f"xAtomRel{i}", f"yAtomRel{i}", f"element{i}"]
        Writer.writerow(row)
        Writer = None

def createTopLineRelative(csvFilePath):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName", "element1", "element2", "element3", "element4", "xAtomRel1", "xAtomRel2", "xAtomRel3", "xAtomRel4", "yAtomRel1", "yAtomRel2", "yAtomRel3", "yAtomRel4"])
        # Writer.writerow(["fileName", "element", "xAtomRel", "yAtomRel"])
        Writer = None

def createTopLinePixels(csvFilePath, XDIMTILES, YDIMTILES, maxPooling = 1):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName"]+ [f"pixelx{x}y{y}" for y in np.arange(XDIMTILES//maxPooling) for x in np.arange(YDIMTILES//maxPooling)])
        Writer = None    


def createTopLineAllPositions(csvFilePath, size, maxPooling = 1):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName"]+ [f"pixelx{x}y{y}" for y in np.arange(size//maxPooling) for x in np.arange(size//maxPooling)])
        Writer = None

def writeAllRows(rows, trainOrTest, XDIMTILES, YDIMTILES, processID = "", createTopRow = None, timeStamp = 0, maxPooling = 1, size = 15):
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

    if createTopRow:
        if pixelOutput:     
            createTopLineAllPositions(csvFilePath, size, maxPooling=maxPooling)
        else: 
            createTopLineCoords(csvFilePath)
    
    
    #if createTopRow: createTopLineRelative(csvFilePath)
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


    XDIMTILES = 10
    YDIMTILES = 10
    maxPooling = 1
    start = (5,5)
    end = (8,8)
    numberOfPositionsInOneAngstrom = 5
    size = int((end[0] - start[0]) * numberOfPositionsInOneAngstrom)
    BFDdiameter = 18 #chosen on the upper end of the BFD diameters (like +4) to have a good margin
    assert(size % maxPooling == 0)
    testDivider = {"train":1, "test":0.25}
    for i in tqdm(range(max(args["iterations"],1)), disable=False, desc = f"Running {args['iterations']} iterations on process {args['id']}"):
        for trainOrTest in ["train", "test"]:
            if trainOrTest not in args["trainOrTest"]:
                continue
            print(f"PID {os.getpid()} on step {i+1} of {trainOrTest}-data at {datetime.datetime.now()}")
            timeStamp = int(str(time()).replace('.', ''))
            rows = saveAllPosDifPatterns(trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, BFDdiameter, processID=args["id"], silence=True, structure = args["structure"], start=start, end=end, maxPooling=maxPooling)
            #rows = saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, BFDdiameter, processID=args["id"], silence=True, maxPooling = maxPooling, structure = args["structure"], start=start, end=end)
            writeAllRows(rows=rows, trainOrTest=trainOrTest, XDIMTILES=XDIMTILES, YDIMTILES=YDIMTILES, processID=args["id"], timeStamp = timeStamp, maxPooling=maxPooling, size = size)
  
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
