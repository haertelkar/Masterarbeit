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
faulthandler.register(signal.SIGUSR1.value)
# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring(seconds_frozen=60, test_interval=100)


device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Calculating on {device}")
xlen = ylen = 5


def moveAndRotateAtomsAndOrthogonalize(atoms, xPos, yPos, zPos, ortho = True) -> Atoms:
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

def grapheneC(xPos = None, yPos = None, zPos = None) -> Atoms:
    try:
        grapheneC = read('structures/graphene.cif')
    except FileNotFoundError:
        grapheneC = read('FullPixelGridML/structures/graphene.cif')
    grapheneC_101 = surface(grapheneC, indices=(1, 0, 1), layers=5, periodic=True)
    grapheneC_slab = moveAndRotateAtomsAndOrthogonalize(grapheneC_101, xPos, yPos, zPos)
    return grapheneC_slab

def MoS2(xPos = None, yPos = None, zPos = None) -> Atoms:
    try:
        molybdenum_sulfur = read('structures/MoS2.cif')
    except FileNotFoundError:
        molybdenum_sulfur = read('FullPixelGridML/structures/MoS2.cif')
    molybdenum_sulfur_011 = surface(molybdenum_sulfur, indices=(0, 1, 1), layers=2, periodic=True)
    molybdenum_sulfur_slab = moveAndRotateAtomsAndOrthogonalize(molybdenum_sulfur_011, xPos, yPos, zPos)
    return molybdenum_sulfur_slab

def Si(xPos = None, yPos = None, zPos = None):
    try:
        silicon = read('structures/Si.cif')
    except FileNotFoundError:
        silicon = read('FullPixelGridML/structures/Si.cif')
    silicon_011 = surface(silicon, indices=(0, 1, 1), layers=2, periodic=True)
    silicon_slab = moveAndRotateAtomsAndOrthogonalize(silicon_011, xPos, yPos, zPos)
    return silicon_slab

def copper(xPos=None, yPos=None, zPos=None) -> Atoms:
    try:
        copper = read('structures/Cu.cif')
    except FileNotFoundError:
        copper = read('FullPixelGridML/structures/Cu.cif')
    copper_111 = surface(copper, indices=(1, 1, 1), layers=2, periodic=True)
    copper_slab = moveAndRotateAtomsAndOrthogonalize(copper_111, xPos, yPos, zPos)
    return copper_slab

def iron(xPos=None, yPos=None, zPos=None) -> Atoms:
    try:
        iron = read('structures/Fe.cif')
    except FileNotFoundError:
        iron = read('FullPixelGridML/structures/Fe.cif')
    iron_111 = surface(iron, indices=(1, 1, 1), layers=2, periodic=True)
    iron_slab = moveAndRotateAtomsAndOrthogonalize(iron_111, xPos, yPos, zPos)
    return iron_slab

def GaAs(xPos = None, yPos = None, zPos = None):
    try:
        gaas = read('structures/GaAs.cif')
    except FileNotFoundError:
        gaas = read('FullPixelGridML/structures/GaAs.cif')
    gaas_110 = surface(gaas, indices=(1, 1, 0), layers=2, periodic=True)
    gaas_slab = moveAndRotateAtomsAndOrthogonalize(gaas_110, xPos, yPos, zPos)
    return gaas_slab

def SrTiO3(xPos = None, yPos = None, zPos = None):
    try:
        srtio3 = read('structures/SrTiO3.cif')
    except FileNotFoundError:
        srtio3 = read('FullPixelGridML/structures/SrTiO3.cif')
    srtio3_110 = surface(srtio3, indices=(1, 1, 0), layers= 5, periodic=True)
    srtio3_slab = moveAndRotateAtomsAndOrthogonalize(srtio3_110, xPos, yPos, zPos)
    return srtio3_slab

def MAPbI3(xPos = None, yPos = None, zPos = None):
    try:
        mapi = read('structures/H6PbCI3N.cif')
    except FileNotFoundError:
        mapi = read('FullPixelGridML/structures/H6PbCI3N.cif')
    mapi_110 = surface(mapi, indices=(1, 1, 0), layers=2, periodic=True)
    mapi_slab = moveAndRotateAtomsAndOrthogonalize(mapi_110, xPos, yPos, zPos)
    return mapi_slab

def WSe2(xPos = None, yPos = None, zPos = None):
    try:
        wse2 = read('structures/WSe2.cif')
    except FileNotFoundError:
        wse2 = read('FullPixelGridML/structures/WSe2.cif')
    wse2_110 = surface(wse2, indices=(1, 1, 0), layers=2, periodic=True)
    wse2_slab = moveAndRotateAtomsAndOrthogonalize(wse2_110, xPos, yPos, zPos)
    return wse2_slab

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
        "copper" : copper,
        "iron" : iron,
        "MarcelsEx" : MarcelsEx,
    }
    #TODO: add more structures
    #TODO: use surface (see Download/ex2.py)
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

@njit
def findAtomsInTile(xPos:float, yPos:float, xRealLength:float, yRealLength:float, atomPositions:np.ndarray):
    xAtomPositions, yAtomPositions, _ = atomPositions.transpose()
    atomPosInside = atomPositions[(xPos <= xAtomPositions) * (xAtomPositions <= xPos + xRealLength) * (yPos <= yAtomPositions) * (yAtomPositions <= yPos + yRealLength)]
    xPositions, yPositions = atomPosInside.transpose()[:2]
    return xPositions, yPositions

def generateDiffractionArray(conv_angle = 33, energy = 60e3, structure = "random", pbar = False, start = (0,0), end = (-1,-1)):

    nameStruct, atomStruct = createStructure(specificStructure= structure)
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

        plt.imsave("difPattern.png", measurement_thick.array[0,0])
        #TODO: add noise
        #TODO: give angle, conv_angle, energy, real pixelsize to ai

        return nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick

def createAllXYCoordinates(yMaxCoord, xMaxCoord):
    return [(x,y) for y in np.arange(yMaxCoord) for x in np.arange(xMaxCoord)]


def generateGroundThruthRelative(rows, atomStruct, datasetStructID, xRealLength, yRealLength, xCoord, yCoord, xPosTile, yPosTile):
    #generates row in labels.csv with relative distances of three closest atoms to the center of the diffraction pattern. Also saves the element predictions.
    
    if len(atomStruct.positions) > 3:
        atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(), atomStruct.get_atomic_numbers(), xPosTile + xRealLength/2, yPosTile + yRealLength/2)
                
    xAtomRel = xPositionsAtoms - xPosTile
    yAtomRel = yPositionsAtoms - yPosTile
    rows.append([f"{datasetStructID}[{xCoord}][{yCoord}]"] + [str(difParams) for difParams in [no for no in atomNumbers] + [x for x in xAtomRel] + [y for y in yAtomRel]])
    return rows

def generateGroundThruthPixel(rows, XDIMTILES, YDIMTILES, atomStruct, datasetStructID, xRealLength, yRealLength, xCoord, yCoord, xPosTile, yPosTile):
    #Generates row in labels.csv with XDIMTILES*YDIMTILES pixels. Each pixel is one if an atom is in the pixel and zero if not.
    pixelGrid = np.zeros((XDIMTILES, YDIMTILES))
    xPositions, yPositions = findAtomsInTile(xPosTile, yPosTile, xRealLength, yRealLength, atomStruct.get_positions())
    xPositions = (xPositions - xPosTile) // 0.2
    yPositions = (yPositions - yPosTile) // 0.2
    pixelGrid[xPositions, yPositions] = 1
    rows.append([f"{datasetStructID}[{xCoord}][{yCoord}]"] + [str(pixel) for pixel in pixelGrid.flatten()])
    return rows

def saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, numberOfPatterns, timeStamp, processID = 99999, silence = False):
    rows = []
    xStepSize = (XDIMTILES - 1)//2
    yStepSize = (YDIMTILES - 1)//2
    allTiles = XDIMTILES * YDIMTILES
    with h5py.File(os.path.join(f"measurements_{trainOrTest}",f"{processID}_{timeStamp}.hdf5"), 'w') as file:
        for cnt, (nameStruct, gridSampling, atomStruct, measurement_thick, _) in enumerate((generateDiffractionArray() for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))):
            datasetStructID = f"{cnt}{processID}{timeStamp}" 
            atomNumbers, xPositionsAtoms, yPositionsAtoms = [],None,None
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

                difPatternsOnePosition = measurement_thick.array[xCNT:xCNT + 2*xStepSize + 1, yCNT :yCNT + 2*yStepSize + 1].copy() # type: ignore
                difPatternsOnePosition = np.reshape(difPatternsOnePosition, (-1,difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))
                difPatternsOnePosition[np.random.choice(difPatternsOnePosition.shape[0], randint(allTiles - 15,allTiles - 5))] = np.zeros((difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))            

                xPosTile = xCNT * gridSampling[0]
                yPosTile = yCNT * gridSampling[1]

                #rows = generateGroundThruthRelative(rows, atomStruct, datasetStructID, xRealLength, yRealLength, xCoord, yCoord, xPosTile, yPosTile)
                rows = generateGroundThruthPixel(rows, XDIMTILES, YDIMTILES, atomStruct, datasetStructID, xRealLength, yRealLength, xCoord, yCoord, xPosTile, yPosTile)

                difPatternsOnePositionResized = []

                for cnt, difPattern in enumerate(difPatternsOnePosition):
                    difPatternsOnePositionResized.append(cv2.resize(np.array(difPattern), dsize=(50, 50), interpolation=cv2.INTER_LINEAR))  # type: ignore
                difPatternsAllPositions[xCoord][yCoord] = np.array(difPatternsOnePositionResized)
                
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