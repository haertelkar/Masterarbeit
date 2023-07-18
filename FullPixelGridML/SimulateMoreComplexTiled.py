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
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1.value)
# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring(seconds_frozen=60, test_interval=100)


device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Calculating on {device}")
xlen = ylen = 5


def moveAtomsAndOrthogonalize(atoms:Atoms, xPos, yPos, zPos, ortho = True) -> Atoms:
    xPos = random()*xlen/3 + xlen/3 if xPos is None else xPos
    yPos = random()*ylen/3 + ylen/3 if yPos is None else yPos
    zPos = 0 if zPos is None else zPos
    atoms.positions += np.array([xPos, yPos, zPos])[None,:]
    if ortho: atoms = orthogonalize_cell(atoms, max_repetitions=10)
    return atoms

def atomPillar(xPos = None, yPos = None, zPos = None, zAtoms = randint(1,10), xAtomShift = 0, yAtomShift = 0, element = None) -> Atoms:
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
    atomPillar = moveAtomsAndOrthogonalize(atomPillar, xPos, yPos, zPos, ortho=False)
    return atomPillar

def grapheneC(xPos = None, yPos = None, zPos = None) -> Atoms:
    grapheneC = graphene(a=2.46,  # Lattice constant (in Angstrom)
                              size=(1, 1, 1))  # Number of unit cells in each direction
    grapheneC = moveAtomsAndOrthogonalize(grapheneC, xPos, yPos, zPos)
    return grapheneC

def MoS2(xPos = None, yPos = None, zPos = None) -> Atoms:
    molybdenum_sulfur = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=None)
    molybdenum_sulfur = moveAtomsAndOrthogonalize(molybdenum_sulfur, xPos, yPos, zPos)
    return molybdenum_sulfur

def Si(xPos = None, yPos = None, zPos = None):
    silicon = bulk('Si', 'diamond', a=5.43, cubic=True)
    silicon = moveAtomsAndOrthogonalize(silicon , xPos, yPos, zPos)
    return silicon

def GaAs(xPos = None, yPos = None, zPos = None):
    gaas = bulk('GaAs', 'zincblende', a=5.65, cubic=True)
    gaas = moveAtomsAndOrthogonalize(gaas , xPos, yPos, zPos)
    return gaas

def SrTiO3(xPos = None, yPos = None, zPos = None):
    srtio3 = read('structures/SrTiO3.cif')
    srtio3 = moveAtomsAndOrthogonalize(srtio3, xPos, yPos, zPos)
    return srtio3

def MAPbI3(xPos = None, yPos = None, zPos = None):
    mapi = read('structures/H6PbCI3N.cif')
    mapi = moveAtomsAndOrthogonalize(mapi, xPos, yPos, zPos)
    return mapi

def WSe2(xPos = None, yPos = None, zPos = None):
    wse2 = read('structures/WSe2.cif')
    wse2 = moveAtomsAndOrthogonalize(wse2, xPos, yPos, zPos)
    return wse2

def StructureUnknown(struct):

    raise Exception(f"Structure {struct} Unknown")

def createStructure(specificStructure : str = "random", **kwargs) -> Atoms:
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
        "atomPillar" : atomPillar
    }

    if specificStructure == "random":
        nameStruct = choice(list(structureFunctions.keys()))
        #tqdm.write(nameStruct)
        struct = structureFunctions.get(nameStruct)(**kwargs)
        return nameStruct, struct
    else:
        nameStruct = specificStructure
        #tqdm.write(nameStruct)
        struct = structureFunctions.get(specificStructure(**kwargs), StructureUnknown(specificStructure))
        return nameStruct, struct 

@njit
def threeClosestAtoms(atomPositions:np.ndarray, atomicNumbers:np.ndarray, xPos:float, yPos:float):
    xyDistances = (atomPositions - np.expand_dims(np.array([xPos, yPos, 0]),0))[:,0:2]
    xyDistances = xyDistances[:,0] + xyDistances[:,1] * 1j
    xyDistanceSortedIndices = np.absolute(xyDistances).argsort() 

    while len(xyDistanceSortedIndices) < 3: #if less than three atoms, just append the closest one again
        xyDistanceSortedIndices += xyDistanceSortedIndices
    
    atomNumbers = atomicNumbers[xyDistanceSortedIndices[:3]]
    xPositions, yPositions = atomPositions[xyDistanceSortedIndices[:3]].transpose()[:2]
    return atomNumbers, xPositions, yPositions

def createSmallTiles(array2D, xDim: int, yDim: int):
    """Creates small 2d tiles from bigger array. The size of the small tiles is given by (xDim, yDim). Any overhang is discarded.

    Args:
        array2D (2D nested iterable): input Array
        xDim (int): x dimension of tiles
        yDim (int): y dimension of tiles

    Raises:
        Exception: xDim and yDim should be smaller than the shape of the array

    Yields:
        same type as array2D: Smaller tiles
    """
    if array2D.shape[0] < xDim or array2D.shape[1] < yDim:
        raise Exception("Tiling dimension are larger than original array.")
    x_ = np.arange(array2D.shape[0]) // xDim #indexes every position to belong to one particular tile
    y_ = np.arange(array2D.shape[1]) // yDim #same in y direction
    indicesOfTilesInX = np.unique(x_)
    indicesOfTilesInY = np.unique(y_)
    #skip the last row/line when less pixel than xDim or yDim
    if len(array2D[x_ == indicesOfTilesInX[-1]]) < xDim:
        indicesOfTilesInX = indicesOfTilesInX[:-1]
    if len(array2D[:,y_ == indicesOfTilesInY[-1]]) < yDim:
        indicesOfTilesInY = indicesOfTilesInY[:-1]
    for x, y in zip(indicesOfTilesInX, indicesOfTilesInY):
        yield x, y, array2D[x_ == x][:, y_ == y]

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

def saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, numberOfPatterns, processID = ""):
    rows = []
    for nameStruct, gridSampling, atomStruct, measurement_thick in (generateDiffractionArray() for i in tqdm(range(numberOfPatterns), leave = False, desc = f"Calculating {trainOrTest}ing data {processID}")):
        if len(atomStruct.positions) <= 3:
            atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(),atomStruct.get_atomic_numbers(), 0, 0)
    
        xRealLength = XDIMTILES * gridSampling[0]
        yRealLength = YDIMTILES * gridSampling[1]

        for xCNT, yCNT, difPatternArray in tqdm(createSmallTiles(measurement_thick.array, XDIMTILES, YDIMTILES), leave=False,desc = f"Going through diffraction Pattern in {XDIMTILES}x{YDIMTILES} tiles {processID}", total= len(measurement_thick.array)):
            xPos = xCNT * gridSampling[0]
            yPos = yCNT * gridSampling[1]
            #findAtomsInTile(xPos, yPos, xRealLength, yRealLength, atomStruct.get_positions())
            if len(atomStruct.positions) > 3:
                atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(), atomStruct.get_atomic_numbers(), xPos + xRealLength/2, yPos + yRealLength/2)
            xAtomRel = xPositionsAtoms - xPos
            yAtomRel = yPositionsAtoms - yPos
            difPatterns = []
            # difPatterns.append(difPatternArray[0,0])
            # difPatterns.append(difPatternArray[0,-1])
            # difPatterns.append(difPatternArray[-1,0])
            # difPatterns.append(difPatternArray[-1,-1])
            for x in range(difPatternArray.shape[0]):
                for y in range(difPatternArray.shape[1]):
                    difPatterns.append(difPatternArray[x][y])
            #difPatterns = np.array(difPatterns)
            for cnt, difPattern in enumerate(difPatterns):
                difPatterns[cnt] = cv2.resize(np.array(difPattern), dsize=(50, 50), interpolation=cv2.INTER_LINEAR)
            fileName = os.path.join(f"measurements_{trainOrTest}",f"{nameStruct}_{xPos}_{yPos}_{np.array2string(atomNumbers)}_{np.array2string(xAtomRel)}_{np.array2string(yAtomRel)}.npy")
            np.save(fileName, np.array(difPatterns))
            rows += [fileName.split(os.sep)[-1]] + [str(difParams) for difParams in [no for no in atomNumbers] + [x for x in xAtomRel] + [y for y in yAtomRel]]
    return rows
                

def createTopLine(trainOrTest, processID = ""):
    with open(os.path.join(f'measurements_{trainOrTest}',f'labels{processID}.csv'), 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName", "element1", "element2", "element3", "xAtomRel1", "xAtomRel2", "xAtomRel3", "yAtomRel1", "yAtomRel2", "yAtomRel3"])
        Writer = None

def writeAllRows(rows, trainOrTest, processID = "", createTopRow = None):
    if createTopRow is None: createTopRow = not os.path.exists(os.path.join(f'measurements_{trainOrTest}',f'labels{processID}.csv'))
    if createTopRow: createTopLine(trainOrTest,processID=processID)
    with open(os.path.join(f'measurements_{trainOrTest}',f'labels{processID}.csv'), 'a', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            Writer.writerow(row)



if __name__ == "__main__":
    print("Running")
    import argparse
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("-id", "--id", type=str, required=False, default= "",help="version number")
    args = vars(ap.parse_args())

    XDIMTILES = 5
    YDIMTILES = 5

    for trainOrTest in ["train", "test"]:
        for i in tqdm(range(74)):
            rows = saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, 20, processID=args["id"])
            writeAllRows(rows=rows, trainOrTest=trainOrTest,processID=args["id"])
        tqdm.write(datetime.datetime.now())

        
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