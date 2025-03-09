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
from abtem import Potential, FrozenPhonons, Probe
from abtem.transfer import CTF, scherzer_defocus, point_resolution, energy2wavelength
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
windowSizeInA = 16 #every 5th A is a scan (probe radius is 5A), should be at least 3 in a window
numberOfAtomsInWindow = windowSizeInA**2
pixelOutput = False

def calc_diameter_bfd(image):
    brightFieldDisk = np.zeros_like(image)
    brightFieldDisk[image > np.max(image)*0.1 + np.min(image)*0.9] = 1
    bfdArea = np.sum(brightFieldDisk)
    diameterBFD = np.sqrt(bfdArea/np.pi) * 2
    return diameterBFD

def moveAndRotateAtomsAndOrthogonalizeAndRepeat(atoms : Atoms, xlen, ylen, xPos = None, yPos = None, zPos = None, ortho = True, repeat = True) -> Atoms:
    if ortho: atoms = orthogonalize_cell(atoms, max_repetitions=10) # type: ignore
    xLength, yLength, zLength = np.max(atoms.positions, axis = 0)
    
    if repeat: atoms_slab = atoms.repeat((int(max(np.ceil(xlen/xLength), 1)), int(max(np.ceil(ylen/yLength),1)), 1))
    else: atoms_slab = atoms
    return atoms_slab # type: ignore

def createAtomPillar(xlen, ylen, xPos = None, yPos = None, zPos = None, zAtoms = randint(1,10), xAtomShift = 0, yAtomShift = 0, element = None) -> Atoms:
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
# def multiPillars(xAtomShift = 0, yAtomShift = 0, element = None, numberOfRandomAtomPillars = None) -> Atoms:
#     numberOfRandomAtomPillars = numberOfRandomAtomPillars or randint(25,50)
#     xPos = random()*ylen 
#     yPos = random()*xlen
#     zPos = 0
#     atomPillar = createAtomPillar(xPos = xPos, yPos = yPos, zPos = zPos)
#     for _ in range(numberOfRandomAtomPillars - 1):
#         xPos = random()*ylen 
#         yPos = random()*xlen
#         atomPillar.extend(createAtomPillar(xPos = xPos, yPos = yPos, zPos = zPos))
#     #atomPillar_011 = surface(atomPillar, indices=(0, 1, 1), layers=2, periodic=True)
#     atomPillar_slab = moveAndRotateAtomsAndOrthogonalizeAndRepeat(atomPillar, xPos, yPos, zPos, ortho=True)
#     return atomPillar_slab

def MarcelsEx(xPos = None, yPos = None, zPos = None):
    cnt1 = nanotube(10, 4, length=4)
    cnt2 = nanotube(21, 0, length=6)
    double_walled_cnt =  cnt1 + cnt2
    double_walled_cnt.rotate(-90, 'x', rotate_cell=True)
    double_walled_cnt.center(vacuum=5, axis=(0,1))
    orthogonal_atoms = moveAndRotateAtomsAndOrthogonalizeAndRepeat(double_walled_cnt,xPos, yPos, zPos, ortho=True, repeat=False)
    return orthogonal_atoms

def grapheneC(xPos = None, yPos = None, zPos = None) -> Atoms:
    grapheneC = graphene()
    grapheneC_slab = moveAndRotateAtomsAndOrthogonalizeAndRepeat(grapheneC, xPos, yPos, zPos)
    return grapheneC_slab

def StructureUnknown(**kwargs):
    raise Exception(f"Structure unknown")

def createStructure(xlen, ylen, specificStructure : str = "random", trainOrTest = None, simple = False, nonPredictedBorderInA = 0,start = (0,0), **kwargs) -> Tuple[str, Atoms]:
    """ Creates a specified structure. If structure is unknown an Exception will be thrown. Default is "random" which randomly picks a structure


    Args:
        specificStructure (str, optional): Give structure. Defaults to "random".

        other arguments: other arguments are possible depending on the structure

    Returns:
        Atoms: Ase Atoms object of specified structure
    """
    if simple:
        nameStruct = "createAtomPillar"
        structFinished = createAtomPillar(xlen = xlen, ylen = ylen, xPos=start[0]+nonPredictedBorderInA+random()*windowSizeInA, yPos=start[1]+nonPredictedBorderInA+random()*windowSizeInA, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100))
        for _ in range(numberOfAtomsInWindow-1):
            structFinished.extend(createAtomPillar(xlen = xlen, ylen = ylen, xPos=start[0]+nonPredictedBorderInA+random()*windowSizeInA, yPos=start[1]+nonPredictedBorderInA+random()*windowSizeInA, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100)))
        
        #Fill the nonPredictedBorderInA with random atoms
        if nonPredictedBorderInA > 0:
            borderArea = (2*nonPredictedBorderInA + windowSizeInA)**2-windowSizeInA**2
            numberOfAtomsInBorder = int(borderArea/windowSizeInA**2 * numberOfAtomsInWindow)
            

            for _ in range(numberOfAtomsInBorder):
                foundPos = False
                while not foundPos:
                    xPosInBorder = start[0]+(2*nonPredictedBorderInA+windowSizeInA)*random()
                    yPosInBorder = start[1]+(2*nonPredictedBorderInA+windowSizeInA)*random()
                    if start[0]+ nonPredictedBorderInA <=xPosInBorder <= start[0]+ nonPredictedBorderInA + windowSizeInA and start[1]+ nonPredictedBorderInA <=yPosInBorder <= start[1]+ nonPredictedBorderInA + windowSizeInA:
                        #in inner square not allowed
                        pass
                    else:
                        foundPos = True
                structFinished.extend(createAtomPillar(xlen = xlen, ylen = ylen, xPos=xPosInBorder, yPos=yPosInBorder, zPos=0, zAtoms=1, xAtomShift=0, yAtomShift=0, element=randint(1,100)))
        
        structFinished = orthogonalize_cell(structFinished, max_repetitions=10)
        return nameStruct, structFinished # type: ignore
    predefinedFunctions = {
        "createAtomPillar" : createAtomPillar,
        #"multiPillars" : multiPillars,
        "MarcelsEx" : MarcelsEx,
        "grapheneC" : grapheneC,
    }
    if specificStructure != "random":
        if ".cif" in specificStructure:
            struct = read(specificStructure)
            nameStruct = specificStructure.split("\\")[-1].split(".")[0]
            structFinished = moveAndRotateAtomsAndOrthogonalizeAndRepeat(struct, xlen, ylen)
        else:
            nameStruct = specificStructure
            structFinished = predefinedFunctions[nameStruct](**kwargs)
    else:
        if os.path.exists(f'FullPixelGridML/structures'):
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
            structFinished = moveAndRotateAtomsAndOrthogonalizeAndRepeat(struct, xlen, ylen) 
    return nameStruct, structFinished

def generateDiffractionArray(trainOrTest = None, conv_angle = 33, energy = 60e3, structure = "random", pbar = False, start = (5,5), end = (20,20), simple = False, nonPredictedBorderInA = 0) -> Tuple[str, Tuple[float, float], Atoms, Measurement, Potential]:
    xlen_structure = end[0] + start[0]
    ylen_structure = end[1] + end[0]

    nameStruct, atomStruct = createStructure(xlen_structure, ylen_structure, specificStructure= structure, trainOrTest = trainOrTest, simple=simple, nonPredictedBorderInA = nonPredictedBorderInA, start=start)
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
        #ctf.defocus = -5 
        #ctf.semiangle_cutoff = 1000 * energy2wavelength(ctf.energy) / point_resolution(Cs, ctf.energy)
        probe = Probe(semiangle_cutoff=conv_angle, energy=energy, defocus = -150, device=device)
        #print(f"FWHM = {probe.profiles().width().compute()} Å")
        probe.match_grid(potential_thick)

        pixelated_detector = PixelatedDetector(max_angle=100,resample = "uniform")

        

        gridSampling = (0.2,0.2)
        if end == (-1,-1):
            end = potential_thick.extent
        gridscan = GridScan(
            start = start, end = end, sampling=gridSampling
        )
        measurement_thick = probe.scan(gridscan, pixelated_detector, potential_thick, pbar = pbar)
        # measurement_thick = poisson_noise(measurement_thick, 1e6) # type: ignore

        # plt.imsave("difPattern.png", measurement_thick.array[0,0])
        #TODO: add noise
        #We dont give angle, conv_angle, energy, real pixelsize to ai because its the same for all training data. Can be done in future

        return nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick # type: ignore

def createAllXYCoordinates(yMaxCoord, xMaxCoord):
    return [(x,y) for y in np.arange(yMaxCoord+1) for x in np.arange(xMaxCoord+1)]
    
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
def generateXYE(datasetStructID, rows, atomStruct, start, end, silence, nonPredictedBorderInA = 0):
    allPositions = atomStruct.get_positions()
    allPositionsShifted = allPositions - np.array([start[0], start[1], 0])
    numberOfPositionsInOneAngstrom = 5
    silence = False
    xMaxCoord, yMaxCoord = int((end[0] - start[0]-nonPredictedBorderInA)* numberOfPositionsInOneAngstrom), int((end[1] - start[1]-nonPredictedBorderInA) * numberOfPositionsInOneAngstrom)
    xMinCoord, yMinCoord = int(nonPredictedBorderInA*numberOfPositionsInOneAngstrom), int(nonPredictedBorderInA*numberOfPositionsInOneAngstrom)
    xyes = []
    cnt = 0
    for (x, y), atomNo in tqdm(zip(allPositionsShifted[:,0:2], atomStruct.get_atomic_numbers()), leave=False, desc = f"Going through diffraction Pattern in atoms.", total= len(allPositionsShifted), disable=True):
        xCoordinate = x*numberOfPositionsInOneAngstrom - xMinCoord
        yCoordinate = y*numberOfPositionsInOneAngstrom - yMinCoord
        
        if xCoordinate <= 0 or yCoordinate <= 0: continue 
        if xCoordinate >= xMaxCoord - xMinCoord  or yCoordinate >= yMaxCoord - yMinCoord : continue
        xyes.append([xCoordinate,yCoordinate,atomNo])
        cnt += 1
    if len(xyes) != numberOfAtomsInWindow:
        print("xyes", xyes)
        raise Exception("Too many/few atoms")
    xyes = np.array(xyes)
    xyes = xyes[xyes[:,0].argsort()] #sort by x coordinate

    rows.append([f"{datasetStructID}"]+list(xyes.flatten().astype(str)))    
    return rows

def saveAllPosDifPatterns(trainOrTest, numberOfPatterns, timeStamp, BFDdiameter,maxPooling = 1, processID = 99999, silence = False, structure = "random", fileWrite = True, difArrays = None,     start = (5,5), end = (8,8), simple = False, nonPredictedBorderInA = 0):
    rows = []
    # rowsPixel = []
    dim = 50

    if fileWrite: file = h5py.File(os.path.join(f"measurements_{trainOrTest}",f"{processID}_{timeStamp}.hdf5"), 'w')
    else: dataArray = []
    difArrays = difArrays or (generateDiffractionArray(trainOrTest = trainOrTest, structure=structure, start=start, end=end, simple = simple, nonPredictedBorderInA=nonPredictedBorderInA) for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))
    for cnt, (nameStruct, gridSampling, atomStruct, measurement_thick, _) in enumerate(difArrays):
        datasetStructID = f"{cnt}{processID}{timeStamp}"         

        difPatternsOnePosition = measurement_thick.array.copy() # type: ignore
        difPatternsOnePosition = np.reshape(difPatternsOnePosition, (-1,difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))
        if pixelOutput:
            rows = generateAtomGridNoInterp(datasetStructID, rows, atomStruct, start, end, silence, maxPooling=maxPooling)
        else:
            rows = generateXYE(datasetStructID, rows, atomStruct, start, end, silence, nonPredictedBorderInA)
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

def createTopLineCoords(csvFilePath):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        row = ["fileName"]
        for i in range(numberOfAtomsInWindow):
            row = row + [f"xAtomRel{i}", f"yAtomRel{i}", f"element{i}"]
        Writer.writerow(row)
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
    start = (0,0)   
    nonPredictedBorderInA = 3
    end = (start[0] + nonPredictedBorderInA * 2 + windowSizeInA , start[1] + nonPredictedBorderInA * 2 + windowSizeInA)
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
            rows = saveAllPosDifPatterns(trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, BFDdiameter, processID=args["id"], silence=True, structure = args["structure"], start=start, end=end, maxPooling=maxPooling, simple = True, nonPredictedBorderInA=nonPredictedBorderInA)
            #rows = saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, BFDdiameter, processID=args["id"], silence=True, maxPooling = maxPooling, structure = args["structure"], start=start, end=end)
            writeAllRows(rows=rows, trainOrTest=trainOrTest, XDIMTILES=XDIMTILES, YDIMTILES=YDIMTILES, processID=args["id"], timeStamp = timeStamp, maxPooling=maxPooling, size = size)
  
    print(f"PID {os.getpid()} done.")

       