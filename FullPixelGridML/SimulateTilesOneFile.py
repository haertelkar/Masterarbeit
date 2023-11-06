from typing import Tuple
from ase import Atoms
from ase.visualize import view
from ase.io import read
from ase.build import surface, mx2, graphene
from abtem.potentials import Potential
import matplotlib.pyplot as plt
from abtem.waves import Probe
import numpy as np
from abtem.measure import block_zeroth_order_spot, Measurement
from abtem.scan import GridScan
from abtem.detect import PixelatedDetector
from random import random, randint, choice
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
from ase.build import molecule, bulk
from time import time
import faulthandler
import signal
from itertools import combinations
import h5py
faulthandler.register(signal.SIGUSR1.value)
# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring(seconds_frozen=60, test_interval=100)


device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Calculating on {device}")
xlen = ylen = 5


def moveAndRotateAtomsAndOrthogonalize(atoms:Atoms, xPos, yPos, zPos, ortho = True) -> Atoms:
    xPos = random()*xlen/3 + xlen/3 if xPos is None else xPos
    yPos = random()*ylen/3 + ylen/3 if yPos is None else yPos
    zPos = 0 if zPos is None else zPos
    atoms.positions += np.array([xPos, yPos, zPos])[None,:]    
    atoms.rotate("x", randint(0,360))
    atoms.rotate("y", randint(0,360))
    atoms.rotate("z", randint(0,360))
    if ortho: atoms = orthogonalize_cell(atoms, max_repetitions=10)
    return atoms

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
    atomPillar = moveAndRotateAtomsAndOrthogonalize(atomPillar, xPos, yPos, zPos, ortho=False)
    return atomPillar

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
    atomPillar = moveAndRotateAtomsAndOrthogonalize(atomPillar, xPos, yPos, zPos, ortho=True)
    return atomPillar

def grapheneC(xPos = None, yPos = None, zPos = None) -> Atoms:
    grapheneC = graphene(a=2.46,  # Lattice constant (in Angstrom)
                              size=(1, 1, 1))  # Number of unit cells in each direction
    num_cells_x = 3  # Number of unit cells in the x-direction
    num_cells_y = 3  # Number of unit cells in the y-direction
    grapheneC *= (num_cells_x, num_cells_y, 1)
    grapheneC = moveAndRotateAtomsAndOrthogonalize(grapheneC, xPos, yPos, zPos)
    return grapheneC

def MoS2(xPos = None, yPos = None, zPos = None) -> Atoms:
    molybdenum_sulfur = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=None)
    molybdenum_sulfur = moveAndRotateAtomsAndOrthogonalize(molybdenum_sulfur, xPos, yPos, zPos)
    return molybdenum_sulfur

def Si(xPos = None, yPos = None, zPos = None):
    silicon = bulk('Si', 'diamond', a=5.43, cubic=True)
    silicon = moveAndRotateAtomsAndOrthogonalize(silicon , xPos, yPos, zPos)
    return silicon

def GaAs(xPos = None, yPos = None, zPos = None):
    gaas = bulk('GaAs', 'zincblende', a=5.65, cubic=True)
    gaas = moveAndRotateAtomsAndOrthogonalize(gaas , xPos, yPos, zPos)
    return gaas

def SrTiO3(xPos = None, yPos = None, zPos = None):
    srtio3 = read('structures/SrTiO3.cif')
    srtio3 = moveAndRotateAtomsAndOrthogonalize(srtio3, xPos, yPos, zPos)
    return srtio3

def MAPbI3(xPos = None, yPos = None, zPos = None):
    mapi = read('structures/H6PbCI3N.cif')
    mapi = moveAndRotateAtomsAndOrthogonalize(mapi, xPos, yPos, zPos)
    return mapi

def WSe2(xPos = None, yPos = None, zPos = None):
    wse2 = read('structures/WSe2.cif')
    wse2 = moveAndRotateAtomsAndOrthogonalize(wse2, xPos, yPos, zPos)
    return wse2

def StructureUnknown(**kwargs):

    raise Exception(f"Structure unknown")

def createStructure(specificStructure : str = "random", **kwargs) -> Tuple[str, Atoms]:
    """ Creates a specified structure. If structure is unknown an Exception will be thrown. Default is "random" which randomly picks a structure


    Args:
        specificStructure (str, optional): Give structure. Defaults to "random".

        other arguments: other arguments are possible depending on the structure

    Returns:
        Atoms: Ase Atoms object of specified structure
    """

    structureFunctions = {
        "graphene" : grapheneC, 
        "MoS2" : MoS2,
        "Si" : Si,
        "GaAs" : GaAs,
        "SrTiO3" : SrTiO3,
        "MAPbI3" : MAPbI3,
        "WSe2" : WSe2,
        "atomPillar" : createAtomPillar,
        "multiPillar" : multiPillars,
    }

    if specificStructure == "random":
        nameStruct = choice(list(structureFunctions.keys()))
        #tqdm.write(nameStruct)
        struct = structureFunctions.get(nameStruct, StructureUnknown)(**kwargs)
        return nameStruct, struct
    else:
        nameStruct = specificStructure
        #tqdm.write(nameStruct)
        struct = structureFunctions.get(specificStructure, StructureUnknown)(**kwargs)
        return nameStruct, struct 

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

# def createSmallTiles(array2D, xDim: int, yDim: int):
#     """Creates small 2d tiles from bigger array. The size of the small tiles is given by (xDim, yDim). Any overhang is discarded.

#     Args:
#         array2D (2D nested iterable): input Array
#         xDim (int): x dimension of tiles
#         yDim (int): y dimension of tiles

#     Raises:
#         Exception: xDim and yDim should be smaller than the shape of the array

#     Yields:
#         same type as array2D: Smaller tiles
#     """
#     if array2D.shape[0] < xDim or array2D.shape[1] < yDim:
#         raise Exception("Tiling dimension are larger than original array.")
#     x_ = np.arange(array2D.shape[0]) // xDim #indexes every position to belong to one particular tile
#     y_ = np.arange(array2D.shape[1]) // yDim #same in y direction
#     indicesOfTilesInX = np.unique(x_)
#     indicesOfTilesInY = np.unique(y_)
#     #skip the last row/line when less pixel than xDim or yDim
#     if len(array2D[x_ == indicesOfTilesInX[-1]]) < xDim:
#         indicesOfTilesInX = indicesOfTilesInX[:-1]
#     if len(array2D[:,y_ == indicesOfTilesInY[-1]]) < yDim:
#         indicesOfTilesInY = indicesOfTilesInY[:-1]
#     for x, y in zip(indicesOfTilesInX, indicesOfTilesInY):
#         yield x, y, array2D[x_ == x][:, y_ == y]

@njit
def findAtomsInTile(xPos:float, yPos:float, xRealLength:float, yRealLength:float, atomPositions:np.ndarray):
    xAtomPositions, yAtomPositions, _ = atomPositions.transpose()
    atomPosInside = atomPositions[(xPos <= xAtomPositions) * (xAtomPositions <= xPos + xRealLength) * (yPos <= yAtomPositions) * (yAtomPositions <= yPos + yRealLength)]
    xPositions, yPositions = atomPosInside.transpose()[:2]

    if len(xPositions) == 0:
        return np.array([-1,-1,-1]), np.array([-1,-1,-1])
    while len(xPositions < 3):
        xPositions = np.append(xPositions, xPositions[0])
        yPositions = np.append(yPositions, yPositions[0])

    return xPositions, yPositions

def generateDiffractionArray():

    nameStruct, atomStruct = createStructure()
    try:
        potential_thick = Potential(
            atomStruct,
            sampling=0.02,
            parametrization="kirkland",
            device=device
        )
    except Exception as e:
        print(nameStruct, atomStruct)
        raise(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = Probe(semiangle_cutoff=24, energy=200e3, device=device)
        probe.match_grid(potential_thick)

        pixelated_detector = PixelatedDetector(max_angle=100)
        gridSampling = (0.2,0.2)
        gridscan = GridScan(
            start = (0, 0), end = potential_thick.extent, sampling=gridSampling
        )
        measurement_thick = probe.scan(gridscan, pixelated_detector, potential_thick, pbar = False)

        return nameStruct, gridSampling, atomStruct, measurement_thick

def createAllXYCoordinates(yMaxCoord, xMaxCoord):
    return [(x,y) for y in np.arange(yMaxCoord) for x in np.arange(xMaxCoord)]

def saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, numberOfPatterns, timeStamp, processID = 99999, silence = False):
    rows = []
    xStepSize = (XDIMTILES - 1)//2
    yStepSize = (YDIMTILES - 1)//2
    with h5py.File(os.path.join(f"measurements_{trainOrTest}",f"{processID}_{timeStamp}.hdf5"), 'w') as file:
        for cnt, (nameStruct, gridSampling, atomStruct, measurement_thick) in enumerate((generateDiffractionArray() for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))):
            datasetStructID = f"{cnt}{processID}{timeStamp}" 
            atomNumbers, xPositionsAtoms, yPositionsAtoms = None,None,None
            if len(atomStruct.positions) <= 3:
                atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(),atomStruct.get_atomic_numbers(), 0, 0)
        
            xRealLength = XDIMTILES * gridSampling[0]
            yRealLength = YDIMTILES * gridSampling[1]

            xMaxCNT, yMaxCNT = np.shape(measurement_thick.array)[:2] # type: ignore
            xMaxCoord = (xMaxCNT-1)//xStepSize - 2
            yMaxCoord = (yMaxCNT-1)//yStepSize - 2

            if xMaxCoord < 1 or yMaxCoord < 1:
                raise Exception(f"xMaxCoord : {xMaxCoord}, yMaxCoord : {yMaxCoord}, struct {nameStruct}, np.shape(measurement_thick.array)[:2] : {np.shape(measurement_thick.array)[:2]}") # type: ignore

            difPatternsAllPositions = np.zeros((xMaxCoord, yMaxCoord, 121, 50, 50))
            for xCoord, yCoord in tqdm(createAllXYCoordinates(yMaxCoord,xMaxCoord), leave=False,desc = f"Going through diffraction Pattern in {XDIMTILES}x{YDIMTILES} tiles {processID}", total= len(measurement_thick.array), disable=silence): # type: ignore
                xCNT = xStepSize * xCoord
                yCNT = yStepSize * yCoord

                #use nine positions (old) -> difPatternsAllPositons has to be a different shape (9, 50, 50)
                #difPatternsOnePosition = (measurement_thick.array[xCNT:xCNT + 2*xStepSize + 1 :xStepSize,yCNT :yCNT + 2*yStepSize + 1:yStepSize]).copy()  # type: ignore
                #use all positions
                difPatternsOnePosition = measurement_thick.array[xCNT:xCNT + 2*xStepSize + 1, yCNT :yCNT + 2*yStepSize + 1].copy()
                difPatternsOnePosition = np.reshape(difPatternsOnePosition, (-1,difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))
                difPatternsOnePosition[np.random.choice(difPatternsOnePosition.shape[0], randint(5,15))] = np.zeros((difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))            

                xPos = xCNT * gridSampling[0]
                yPos = yCNT * gridSampling[1]

                if len(atomStruct.positions) > 3:
                    atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(), atomStruct.get_atomic_numbers(), xPos + xRealLength/2, yPos + yRealLength/2)
                
                xAtomRel = xPositionsAtoms - xPos
                yAtomRel = yPositionsAtoms - yPos

                difPatternsOnePositionResized = []

                for cnt, difPattern in enumerate(difPatternsOnePosition):
                    difPatternsOnePositionResized.append(cv2.resize(np.array(difPattern), dsize=(50, 50), interpolation=cv2.INTER_LINEAR))  # type: ignore
                difPatternsAllPositions[xCoord][yCoord] = np.array(difPatternsOnePositionResized)
                rows.append([f"{datasetStructID}[{xCoord}][{yCoord}]"] + [str(difParams) for difParams in [no for no in atomNumbers] + [x for x in xAtomRel] + [y for y in yAtomRel]])
            datasetStruct = file.create_dataset(f"{datasetStructID}",data = difPatternsAllPositions, compression="gzip")
            datasetStruct[:] = difPatternsAllPositions
    return rows
                

def createTopLine(csvFilePath):
    with open(csvFilePath, 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName", "element1", "element2", "element3", "xAtomRel1", "xAtomRel2", "xAtomRel3", "yAtomRel1", "yAtomRel2", "yAtomRel3"])
        Writer = None

def writeAllRows(rows, trainOrTest, processID = "", createTopRow = None, timeStamp = 0):
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
    if createTopRow: createTopLine(csvFilePath)
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
    args = vars(ap.parse_args())


    XDIMTILES = 11
    YDIMTILES = 11
    testDivider = {"train":1, "test":0.25}
    for i in tqdm(range(max(args["iterations"],1)), disable=True):
        for trainOrTest in ["train", "test"]:
            if trainOrTest not in args["trainOrTest"]:
                continue
            print(f"PID {os.getpid()} on step {i+1} at {datetime.datetime.now()}")
            timeStamp = int(str(time()).replace('.', ''))
            rows = saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, processID=args["id"], silence=True)
            writeAllRows(rows=rows, trainOrTest=trainOrTest,processID=args["id"], timeStamp = timeStamp)
  
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