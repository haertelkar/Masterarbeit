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
import warnings
import os
from numba import njit
from ase import Atoms
from ase.build import molecule, bulk


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
        "graphene" : grapheneC(**kwargs),
        "MoS2" : MoS2(**kwargs),
        "Si" : Si(**kwargs),
        "GaAs" : GaAs(**kwargs),
        "SrTiO3" : SrTiO3(**kwargs),
        "MAPbI3" : MAPbI3(**kwargs),
        "WSe2" : WSe2(**kwargs),
        "atomPillar" : atomPillar(**kwargs)
    }

    if specificStructure == "random":
        nameStruct = choice(list(structureFunctions.keys()))
        struct = structureFunctions.get(nameStruct)
        return nameStruct, struct
    else:
        nameStruct = specificStructure
        struct = structureFunctions.get(specificStructure, StructureUnknown(specificStructure))
        return nameStruct, struct 

@njit
def threeClosestAtoms(atomPositions:np.ndarray, atomicNumbers:np.ndarray, xPos:float, yPos:float):
    xyDistances = (atomPositions - np.expand_dims(np.array([xPos, yPos, 0]),0))[:,0:2]
    xyDistances = xyDistances[:,0] + xyDistances[:,1] * 1j
    xyDistanceSortedIndices = np.absolute(xyDistances).argsort() #Quatsch, ich muss auch noch die Indices rausfinden

    while len(xyDistanceSortedIndices) < 3: #if less than three atoms, just append the closest one again
        xyDistanceSortedIndices += xyDistanceSortedIndices
    
    atomNumbers = atomicNumbers[xyDistanceSortedIndices[:3]]
    xPositions, yPositions = atomPositions[xyDistanceSortedIndices[:3]].transpose()[:2]
    return atomNumbers, xPositions, yPositions


for trainOrTest in ["train", "test"]:
    with open(os.path.join(f'measurements_{trainOrTest}','labels.csv'), 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName", "element", "xAtomRel", "xAtomShift", "yAtomRel", "yAtomShift", "zAtoms"])
        Writer = None
    for i in tqdm(range(20), desc = f"Calculating {trainOrTest}ing data"):
        nameStruct, atomStruct = createStructure()
        # view(atomPillar)
        # plt.show()
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

            pixelated_detector = PixelatedDetector(max_angle=120)
            coarseSamplingFactor = 2 #so less pixels positions get sampled
            numberOfScanPos = int(np.array(potential_thick.extent)/0.2)
            gridSampling = np.array(potential_thick.extent)/numberOfScanPos
            gridscan = GridScan(
                start = (0, 0), end = potential_thick.extent, gpts=numberOfScanPos
            )
            measurement_thick = probe.scan(gridscan, pixelated_detector, potential_thick, pbar = False)
        

        if len(atomStruct.positions) <= 3:
            atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(),atomStruct.get_atomic_numbers(), 0, 0)

        with open(os.path.join(f'measurements_{trainOrTest}','labels.csv'), 'a', newline='') as csvfile:
            Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for xCNT, difPatternRow in tqdm(enumerate(measurement_thick.array), leave = False, desc="Going through the x scan positions", total= len(measurement_thick.array)):
                for yCNT, difPattern in tqdm(enumerate(difPatternRow), leave = False, desc="Going through the y scan positions", total= len(difPatternRow)):
                    if xCNT %2 == 0 or yCNT %2 == 0: #reduce amount of data from one sample
                        continue
                    xPos = xCNT * gridSampling[0]
                    yPos = yCNT * gridSampling[1]
                    if len(atomStruct.positions) > 3:
                        atomNumbers, xPositionsAtoms, yPositionsAtoms = threeClosestAtoms(atomStruct.get_positions(), atomStruct.get_atomic_numbers(), xPos, yPos)
                    xAtomRel = xPositionsAtoms - xPos
                    yAtomRel = yPositionsAtoms - yPos
                    fileName = os.path.join(f"measurements_{trainOrTest}",f"{nameStruct}_{xPos}_{yPos}_{np.array2string(atomNumbers)}_{np.array2string(xAtomRel)}_{np.array2string(yAtomRel)}.npy")
                    np.save(fileName,difPattern)
                    Writer.writerow([fileName.split(os.sep)[-1]] + [str(difParams) for difParams in [no for no in atomNumbers] + [x for x in xAtomRel] + [y for y in yAtomRel]])
            
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