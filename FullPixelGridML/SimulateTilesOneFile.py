import glob
from typing import Tuple
from ase import Atoms
from ase.visualize import view
from ase.io import read
from ase.build import surface, mx2, graphene
from ase.geometry import get_distances
# from abtem.potentials import Potential
import dask
import matplotlib.pyplot as plt
from abtem.waves import Probe
import numpy as np
from abtem.scan import GridScan
from random import random, randint, choice, uniform, randrange
from abtem.reconstruct import MultislicePtychographicOperator, RegularizedPtychographicOperator
from abtem import Potential, FrozenPhonons, Probe
from abtem.transfer import CTF, scherzer_defocus, point_resolution, energy2wavelength
from abtem.scan import GridScan
from abtem import orthogonalize_cell, PixelatedDetector, GridScan
from abtem.measurements import BaseMeasurements, DiffractionPatterns
import nvidia_smi



def get_gpu_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total memory: {info.total / (1024 ** 2)} MB, Free memory: {info.free / (1024 ** 2)} MB, Used memory: {info.used / (1024 ** 2)} MB")
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
faulthandler.register(signal.SIGUSR1.value)
import sys

# setting path


# from dask_mpi import initialize
# initialize()

# from dask.distributed import Client
# client = Client()
# abtem.config.set({"dask.chunk-size-gpu" : "512 MB"})
# abtem.config.set({"device": "gpu"})
# from hanging_threads import start_monitoring
# monitoring_thread = start_monitoring(seconds_frozen=60, test_interval=100)
# dask.config.set({"num_workers": 1})
# abtem.config.set({"cupy.fft-cache-size" : "1024 MB"})
# device = "gpu"#"gpu" if torch.cuda.is_available() else "cpu"

sys.path.append("/data/scratch/haertelk/Masterarbeit/Zernike")
from ZernikePolynomials import Zernike 
windowSizeInA = 3 # Size of the window in Angstroms
numberOfAtomsInWindow = windowSizeInA**2
pixelOutput = False
FolderAppendix = "_4to8s_-50def_15B_new"#"_4sparse_noEB_-50def_20Z"


def calc_diameter_bfd(image):
    brightFieldDisk = np.zeros_like(image)
    brightFieldDisk[image > np.max(image)*0.1 + np.min(image)*0.9] = 1
    bfdArea = np.sum(brightFieldDisk)
    diameterBFD = np.sqrt(bfdArea/np.pi) * 2
    return diameterBFD

def emptySpace(xlen, ylen, zlen = 1, xPos = None, yPos = None, zPos = None) -> Atoms:
    """Creates an empty Atoms object with the given dimensions."""
    return Atoms(cell=[xlen, ylen, zlen, 90, 90, 90], pbc=True, positions=[[xPos or 0, yPos or 0, zPos or 0]])

def moveAndRotateAtomsAndOrthogonalizeAndRepeat(atoms : Atoms, xlen, ylen, xPos = None, yPos = None, zPos = None, ortho = True, repeat = True) -> Atoms:
    if ortho: atoms = orthogonalize_cell(atoms, max_repetitions=10) # type: ignore
    xLength, yLength, zLength = np.max(atoms.positions, axis = 0)
    
    if repeat: atoms_slab = atoms.repeat((int(max(np.ceil(xlen/xLength), 1)), int(max(np.ceil(ylen/yLength),1)), 1))
    else: atoms_slab = atoms
    return atoms_slab # type: ignore

def createAtomPillar(xlen, ylen, xPos = None, yPos = None, zPos = None, zAtoms = randint(1,10), xAtomShift :float = 0, yAtomShift :float= 0, element = None) -> Atoms:
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
def multiPillars(xlen, ylen, xAtomShift = 0, yAtomShift = 0, element = None, numberOfRandomAtomPillars = None) -> Atoms:
    numberOfRandomAtomPillars = numberOfRandomAtomPillars or randint(25,50)
    xPos = random()*ylen 
    yPos = random()*xlen
    zPos = 0
    atomPillar = createAtomPillar(xlen, ylen, xPos = xPos, yPos = yPos, zPos = zPos)
    for _ in range(numberOfRandomAtomPillars - 1):
        xPos = random()*ylen 
        yPos = random()*xlen
        atomPillar.extend(createAtomPillar(xlen, ylen,xPos = xPos, yPos = yPos, zPos = zPos, xAtomShift = np.random.random()-0.5, yAtomShift = np.random.random()-0.5))
    #atomPillar_011 = surface(atomPillar, indices=(0, 1, 1), layers=2, periodic=True)
    atomPillar_slab = moveAndRotateAtomsAndOrthogonalizeAndRepeat(atomPillar, xPos, yPos, zPos, ortho=True)
    return atomPillar_slab

def MarcelsEx(xlen = None, ylen = None, xPos = None, yPos = None, zPos = None):
    cnt1 = nanotube(10, 4, length=4)
    cnt2 = nanotube(21, 0, length=6)
    double_walled_cnt =  cnt1 + cnt2
    double_walled_cnt.rotate(-90, 'x', rotate_cell=True)
    double_walled_cnt.center(vacuum=5, axis=(0,1))
    orthogonal_atoms = moveAndRotateAtomsAndOrthogonalizeAndRepeat(double_walled_cnt,xPos, yPos, zPos, ortho=True, repeat=False)
    return orthogonal_atoms

def grapheneC(xlen = None, ylen = None, xPos = None, yPos = None, zPos = None) -> Atoms:
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
        "multiPillars" : multiPillars,
        "MarcelsEx" : MarcelsEx,
        "grapheneC" : grapheneC,
        "emptySpace" : emptySpace,
    }
    if specificStructure != "random":
        if ".cif" in specificStructure:
            struct = read(specificStructure)
            nameStruct = specificStructure.split("\\")[-1].split(".")[0]
            structFinished = moveAndRotateAtomsAndOrthogonalizeAndRepeat(struct, xlen, ylen)
        else:
            nameStruct = specificStructure
            structFinished = predefinedFunctions[nameStruct](xlen, ylen, **kwargs)
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
        # print(f"Created structure {nameStruct} with shape {structFinished.get_positions().shape}")
    return nameStruct, structFinished

def generateDiffractionArray(trainOrTest = None, conv_angle = 33, energy = 60e3, 
                             structure = "random", pbar = False, 
                             start = (5,5), end = (20,20), simple = False,
                             nonPredictedBorderInA = 0, device = "gpu",
                             deviceAfter = "gpu", generate_graphics = False, defocus = -50) -> Tuple[str, Tuple[float, float], Atoms, BaseMeasurements, Potential]:
    xlen_structure = start[0] + end[0]
    ylen_structure = start[1] + end[1] 
    xlen_structure = ylen_structure = max(xlen_structure, ylen_structure) #make sure its square
    # print(f"Generating structure with xlen = {xlen_structure} and ylen = {ylen_structure} and start = {start} and end = {end} and nonPredictedBorderInA = {nonPredictedBorderInA}")
    # print(f"Calculating on {device}")
    if structure == "emptySpace":
        nameStruct = "emptySpace"
        atomStruct = np.zeros(grid.shape)
    else:
        nameStruct, atomStruct = createStructure(xlen_structure, ylen_structure, specificStructure= structure, trainOrTest = trainOrTest, simple=simple, nonPredictedBorderInA = nonPredictedBorderInA, start=start)
    try:
        potential_thick = Potential(
            atomStruct,
            device=device,
            sampling=0.05 
        )
    except Exception as e:
        print(nameStruct, atomStruct)
        raise(e)
    # abtem.show_atoms(atomStruct, legend = True)
    # plt.savefig("testStructureOriginal.png")
    # plt.close()
    # exit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #ctf.defocus = -5 
        #ctf.semiangle_cutoff = 1000 * energy2wavelength(ctf.energy) / point_resolution(Cs, ctf.energy)
        probe = Probe(semiangle_cutoff=conv_angle, energy=energy, defocus = defocus, device=device)
        #print(f"FWHM = {probe.profiles().width().compute()} Å")
        probe.match_grid(potential_thick)
        

        pixelated_detector = PixelatedDetector(max_angle=100, resample="uniform")

        #object increase field of view
        #show exit wave before diffraction pattern
        #show probe
        gridSampling = (0.2,0.2)
        if end == (-1,-1):
            end : tuple[float,float]= potential_thick.extent # type: ignore
        gridscan = GridScan(
            start = start, end = end, sampling=gridSampling
        )
        if generate_graphics:
            waves = probe.build(lazy = False)
            print(f"Probe shape: {waves.shape}")
            print(f"waves extent in A: {waves.extent}")
            print(f"BFD diameter in Pixel: {calc_diameter_bfd(waves.intensity().array)}")
            print(f"BFD diameter in A: {calc_diameter_bfd(waves.intensity().array) * waves.extent[0] / waves.shape[0]}")
            #print the lines from above to a file
            with open(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/probeInfo_{defocus}.txt", "w") as f:
                f.write(f"Probe shape: {waves.shape}\n")
                f.write(f"waves extent in A: {waves.extent}\n")
                f.write(f"BFD diameter in Pixel: {calc_diameter_bfd(waves.intensity().array)}\n")
                f.write(f"BFD diameter in A: {calc_diameter_bfd(waves.intensity().array) * waves.extent[0] / waves.shape[0]}\n")
            waves.intensity().show(
            explode=True, cbar=True, common_color_scale=True, figsize=(10, 10)
            )
            plt.savefig(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/testProbe_{defocus}.png")
            plt.close()
            waves.diffraction_patterns().show(explode=True, cbar=True, common_color_scale=True, figsize=(10, 10))
            plt.savefig(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/testProbe_DifPattern_{defocus}.png")
            plt.close()
            waves.diffraction_patterns(block_direct=True).show(explode=True, cbar=True, common_color_scale=True, figsize=(10, 10))
            plt.savefig(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/testProbe_DifPattern_{defocus}.png")
            plt.close()

        
        measurement_thick = probe.scan(potential_thick, gridscan, pixelated_detector)
        if generate_graphics:
            print(f"Measurement shape: {measurement_thick.shape}")
            print(f"structure: {nameStruct}") 
            single_diffraction_pattern = measurement_thick[1, 1]

            abtem.stack(
                [
                    single_diffraction_pattern,
                    single_diffraction_pattern.block_direct(),
                ],
                ("base", "block direct"),
            ).show(explode=True, cbar=True, figsize=(13, 4))
            plt.savefig(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/diffractionPattern_{defocus}.png")
            plt.close()
            plt.imsave(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/difPattern_asis_{defocus}.png", measurement_thick.array[1,1])
            single_diffraction_pattern.show(power=0.2, cbar=True)
            plt.savefig(f"/data/scratch/haertelk/Masterarbeit/SimulationImages/diffractionPattern_{defocus}_powerScaling.png")
            plt.close()
        #measurement_thick : BaseMeasurements= measurement_thick.poisson_noise( 1e5) # type: ignore
        if deviceAfter == "gpu":
            pass
            # measurement_thick = measurement_thick.compute()
        elif deviceAfter == "cpu":
            measurement_thick = measurement_thick.to_cpu()


        # plt.imsave("difPattern.png", measurement_thick.array[0,0])
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


def generateXYE(datasetStructID, rows, atomStruct, start, end, silence = True, nonPredictedBorderInA = 0):
    """
    Generates a list of x, y, and element number of the atoms in the given atom structure.
    Args:
        datasetStructID (str): The ID of the dataset in the hdf5 file.
        rows (list): The list to append the data to.
        atomStruct (Atoms): The atom structure.
        start (tuple): The starting coordinates.
        end (tuple): The ending coordinates.
        silence (bool): If True, suppresses progress bar.
        nonPredictedBorderInA (int, optional): The non-predicted border in Angstroms. Defaults to 0.
    Returns:
        list: The updated rows list, where the entry for this structure has been appended.
    """
    allPositions = atomStruct.get_positions()
    allPositionsShifted = allPositions - np.array([start[0], start[1], 0])
    numberOfPositionsInOneAngstrom = 5
    xMaxCoord, yMaxCoord = int((end[0] - start[0]-nonPredictedBorderInA)* numberOfPositionsInOneAngstrom), int((end[1] - start[1]-nonPredictedBorderInA) * numberOfPositionsInOneAngstrom)
    xMinCoord, yMinCoord = int(nonPredictedBorderInA*numberOfPositionsInOneAngstrom), int(nonPredictedBorderInA*numberOfPositionsInOneAngstrom)
    xyes = []
    cnt = 0
    for (x, y), atomNo in tqdm(zip(allPositionsShifted[:,0:2], atomStruct.get_atomic_numbers()), leave=False, desc = f"Going through diffraction Pattern in atoms.", total= len(allPositionsShifted), disable=silence):
        xCoordinate = x*numberOfPositionsInOneAngstrom - xMinCoord
        yCoordinate = y*numberOfPositionsInOneAngstrom - yMinCoord
        
        if xCoordinate <= 0 or yCoordinate <= 0: continue 
        if xCoordinate >= xMaxCoord - xMinCoord  or yCoordinate >= yMaxCoord - yMinCoord : continue
        xyes.append([xCoordinate,yCoordinate,atomNo])
        cnt += 1
    # print(f"Found {len(xyes)} atoms with nonPredictedBorderInA = {nonPredictedBorderInA} and start = {start} and end = {end}\n")
    xyes = np.array(xyes)
    if len(xyes): xyes = xyes[xyes[:,0].argsort()] #sort by x coordinate

    rows.append([f"{datasetStructID}"]+list(xyes.flatten().astype(str)))    
    return rows

def generate_sparse_grid(x_size, y_size, s, xStartShift = 0, yStartShift = 0, xEndShift = 0, yEndShift = 0, xOffset = 0, yOffset = 0, twoD = False):
    """
    Generates a 1D array of flattened sparse grid coordinates.

    Parameters:
        x_size (int): X size.
        y_size (int): Y size.
        s (int): Sparseness level (step size between coordinates).

    Returns:
        List of coordinates representing flattened sparse grid coordinates.
    """
    assert(xStartShift >= 0)
    assert(yStartShift >= 0)
    assert(xEndShift <= 0)
    assert(yEndShift <= 0)
    if twoD:
        return [(x, y) for x in range(xStartShift + xOffset, x_size + xEndShift, s) for y in range(yStartShift + yOffset, y_size + yEndShift, s)]
    return [x * y_size + y for x in range(xStartShift + xOffset, x_size + xEndShift, s) for y in range(yStartShift + yOffset, y_size + yEndShift, s)]

def saveAllPosDifPatterns(trainOrTest, numberOfPatterns, timeStamp, BFDdiameter,maxPooling = 1, processID = 99999, silence = False, 
                          structure = "random", fileWrite = True, difArrays = None,     start = (5,5), end = (8,8), simple = False, 
                          nonPredictedBorderInA = 0, zernike = False, initialCoords = None, defocus=-50, numberOfOSAANSIMoments = 20, step_size = 10):
    """
    Generates all diffraction patterns for the given positions and saves them to a file.
    Args:
        trainOrTest (str): Indicates whether the data is saved in measurement_train or measurements_test.
        numberOfPatterns (int): The number of diffraction patterns to generate.
        timeStamp (int): A timestamp to be used in the file name.
        BFDdiameter (int): The diameter of the bright field disk.
        maxPooling (int, optional): The maximum pooling size. Defaults to 1.
        processID (int, optional): An identifier for the process. Defaults to 99999.
        silence (bool, optional): If True, suppresses progress bar. Defaults to False.
        structure (str, optional): The structure type. Defaults to "random".
        fileWrite (bool, optional): If True, writes the data to a file. Else it's output as the second return value. Defaults to True.
        difArrays (list, optional): Takes in a list of diffraction arrays instead of calculating new ones. Defaults to None.
        start (tuple, optional): The starting coordinates. Defaults to (5,5).
        end (tuple, optional): The ending coordinates. Defaults to (8,8).
        simple (bool, optional): If True, uses a simple structure. Defaults to False.
        nonPredictedBorderInA (int, optional): The non-predicted border in Angstroms. Defaults to 0.
        zernike (bool, optional): If True, uses Zernike polynomials. Defaults to False.
        positions (list, optional): A list of positions to use. Or use "calculate" to calculate positions. Defaults to None.
    Returns:
        list: A list of rows containing the generated diffraction patterns.
        optional if fileWrite is False: numpy array of the generated diffraction patterns
    """
    rows = []
    # rowsPixel = []
    dimOrig = 50
    dimNew = 0
    BFDdiameterScaled = None
    nonPredictedBorderInCoords = nonPredictedBorderInA * 5
    windowLengthinCoords = windowSizeInA * 5

    if fileWrite: 
        if not zernike:
            fileName = os.path.join(f"measurements_{trainOrTest}{FolderAppendix}",f"{processID}_{timeStamp}.hdf5")
        else:
            fileName = os.path.join("..","Zernike",f"measurements_{trainOrTest}{FolderAppendix}",f"{processID}_{timeStamp}.hdf5")
        file = h5py.File(fileName, 'w', libver='latest')
    else: 
        dataArray = []
    if zernike:
        ZernikeObject = Zernike(numberOfOSAANSIMoments= numberOfOSAANSIMoments)
    else:   
        dimNew = 100   
    


    difArrays = difArrays or (generateDiffractionArray(trainOrTest = trainOrTest, structure=structure, start=start, end=end, simple = simple, nonPredictedBorderInA=nonPredictedBorderInA, defocus=defocus) for i in tqdm(range(numberOfPatterns), leave = False, disable=silence, desc = f"Calculating {trainOrTest}ing data {processID}"))
    for cnt, (_, _, atomStruct, measurement_thick, _) in enumerate(difArrays):
        datasetStructID = f"{cnt}{processID}{timeStamp}"         

        difPatterns = measurement_thick.array#.copy() # type: ignore
        if zernike: dimNew = min(difPatterns.shape[-2],  difPatterns.shape[-1]) 
        BFDdiameterScaled = int(BFDdiameter * dimNew / dimOrig)
        difPatterns = np.reshape(difPatterns, (-1,difPatterns.shape[-2], difPatterns.shape[-1]))
        # print(measurement_thick.array.shape)
        # exit()
        if initialCoords is None:
            sparseGridFactor = randint(4,8)
            MaxShift = nonPredictedBorderInCoords + windowLengthinCoords//2  
            xShift = randint(-MaxShift, MaxShift)
            yShift = randint(-MaxShift, MaxShift)
            xStartShift = max(0, xShift)
            yStartShift = max(0, yShift)
            xEndShift = min(xShift, 0 )
            yEndShift = min(yShift, 0)
            xOffset = randint(0,sparseGridFactor-1)
            yOffset = randint(0,sparseGridFactor-1)
            choosenCoords : np.ndarray = np.array(generate_sparse_grid(measurement_thick.array.shape[0], measurement_thick.array.shape[1],
                                                                        sparseGridFactor, xStartShift=xStartShift, yStartShift=yStartShift,
                                                                        xEndShift=xEndShift,yEndShift=yEndShift, xOffset = xOffset, 
                                                                        yOffset = yOffset, twoD=False)) # type: ignore
        else:
            choosenCoords = initialCoords                                                                                                        
        # print(f"choosenCoords shape: {choosenCoords.shape}, choosenCoords: {choosenCoords}")
        splitting = min(step_size, len(choosenCoords)) #split the calculation into smaller chunks to avoid memory issues
        difPatternsComputed = np.empty((len(choosenCoords), difPatterns.shape[-2], difPatterns.shape[-1]), dtype=difPatterns.dtype) # type: ignore
        for cnt in range((len(choosenCoords) // splitting) + 1):
            chosenCoordsSplit = choosenCoords[cnt*splitting:min((cnt+1)*splitting, len(choosenCoords))]
            
            if len(chosenCoordsSplit) == 0: continue
            if not silence:
                get_gpu_memory()
                print(f"Calculating {cnt * splitting} to {min((cnt + 1) * splitting, len(choosenCoords))} of  {len(choosenCoords)} coordinates")
            difPatternsComputed[cnt*splitting:min((cnt+1)*splitting, len(choosenCoords))] = difPatterns[chosenCoordsSplit].compute(progress_bar = not silence)
        if pixelOutput:
            rows = generateAtomGridNoInterp(datasetStructID, rows, atomStruct, start, end, silence, maxPooling=maxPooling)
        else:
            rows = generateXYE(datasetStructID, rows, atomStruct, start, end, silence, nonPredictedBorderInA)

        difPatternsResized = []#[np.zeros((dimNew, dimNew))] #empty array for the first element, works as csl token. IS PROBLEMATIC because the model than expects this in prediction
        for cnt, difPattern in enumerate(difPatternsComputed): 
            difPatternsResized.append(cv2.resize(difPattern, dsize=(dimNew, dimNew), interpolation=cv2.INTER_LINEAR))  # type: ignore
        difPatternsResized = np.array(difPatternsResized)
        #     difPatternsOnePositionResized.append(difPattern)
        # difPatternsOnePositionResized = np.array(difPatternsOnePositionResized)
        # removing everything outside the bright field disk
        indicesInBFD = slice(max((dimNew - BFDdiameterScaled)//2-1,0),min((dimNew + BFDdiameterScaled)//2+1, dimNew ))
        difPatternsResized = difPatternsResized[:,indicesInBFD, indicesInBFD] 
        
        # plt.imsave(os.path.join(f"measurements_{trainOrTest}",f"{datasetStructID}.png"), difPatternsOnePositionResized[0])
        
        if fileWrite: 
            #TODO x and y are incorrect, should be switched
            choosenXCoords = (choosenCoords % measurement_thick.array.shape[1]).astype(int) 
            choosenYCoords = (choosenCoords / measurement_thick.array.shape[1]).astype(int)
            # choosenCoords2D : np.ndarray = np.array(generate_sparse_grid(measurement_thick.array.shape[0], measurement_thick.array.shape[1], sparseGridFactor, xStartShift=xStartShift,
            #                                                              yStartShift=yStartShift, xEndShift=xEndShift,yEndShift=yEndShift, twoD=True))
            # for cnt, (xCoord, yCoord) in enumerate(choosenCoords2D):
            #     if xCoord != choosenXCoords[cnt] or yCoord != choosenYCoords[cnt]:
            #         raise Exception(f"choosenCoords2D and choosenCoords do not match at index {cnt}: {choosenCoords2D[cnt]} != {choosenXCoords[cnt], choosenYCoords[cnt]}")
            if not zernike:
                difPatternsResized_reshaped_with_Coords = np.concatenate((difPatternsResized.reshape(difPatternsResized.shape[0],-1), np.stack([choosenXCoords - nonPredictedBorderInCoords, choosenYCoords - nonPredictedBorderInCoords]).T), axis = 1)
                file.create_dataset(f"{datasetStructID}", data = difPatternsResized_reshaped_with_Coords.astype('float32'), compression="lzf", chunks = (1, difPatterns.shape[-1]), shuffle = True)
            else:
                zernDifPatterns = ZernikeObject.zernikeTransform(dataSetName = None, groupOfPatterns = difPatternsResized, hdf5File = None)
                padding = np.zeros_like(choosenXCoords)
                zernDifPatterns = np.concatenate((zernDifPatterns, np.stack([choosenXCoords - nonPredictedBorderInCoords, choosenYCoords - nonPredictedBorderInCoords, padding]).T), axis = 1)
                
                file.create_dataset(f"{datasetStructID}", data = zernDifPatterns.astype('float32'), compression="lzf", chunks = (1, zernDifPatterns.shape[-1]), shuffle = True)
            
        else: 
            if not zernike:
                dataArray.append(difPatternsResized)
            else:
                # plt.imsave("/data/scratch/haertelk/Masterarbeit/SimulationImages/difPattern_resized_-150.png", difPatternsResized[15])
                zernDifPatterns = ZernikeObject.zernikeTransform(dataSetName = None, groupOfPatterns = difPatternsResized, hdf5File = None)
                dataArray.append(zernDifPatterns)
                #IMPORTANT: The coords are not appended here. This is done later during reconstruction
    if fileWrite: 
        file.close()
        return rows
    else:
        return rows, np.array(dataArray) # type: ignore

# def calc_diameter_bfd_simple(image):
#     leftEdge = 0
#     rightEdge = 0
#     plt.imsave("testMask.png",(image > (np.max(image)*0.1 + np.min(np.abs(image))*0.9)).get().astype(int))
#     for row in tqdm((image > (np.max(image)*0.1 + np.min(np.abs(image))*0.9)).get().astype(int)):
#         for i in range(len(row)):
#             if row[i] == 1 and i < leftEdge:
#                 leftEdge = i
#             if row[i] == 1 and i > rightEdge:
#                 rightEdge = i

#     if rightEdge < leftEdge:
#         raise Exception("rightEdge < leftEdge")
        
#     diameterBFD = rightEdge - leftEdge
#     return diameterBFD


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

def writeAllRows(rows, trainOrTest, XDIMTILES, YDIMTILES, processID = "", createTopRow = None, timeStamp = 0, maxPooling = 1, size = 15, zernike = False):
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
    csvFilePath = os.path.join(f'measurements_{trainOrTest}{FolderAppendix}',f'labels_{processID}_{timeStamp}.csv')
    if zernike: 
        csvFilePath = os.path.join("..","Zernike",csvFilePath)
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
    print(f"FolderAppendix: {FolderAppendix}")
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
    zernike = False
    end = (start[0] + nonPredictedBorderInA * 2 + windowSizeInA , start[1] + nonPredictedBorderInA * 2 + windowSizeInA)

    numberOfPositionsInOneAngstrom = 5
    defocus = -50 #in Angstroms
    size = int((end[0] - start[0]) * numberOfPositionsInOneAngstrom)
    BFDdiameter = 18 #chosen on the upper end of the BFD diameters (like +4) to have a good margin
    assert(size % maxPooling == 0)
    testDivider = {"train":1, "test":0.05}
    for i in tqdm(range(max(args["iterations"],1)), disable=False, desc = f"Running {args['iterations']} iterations on process {args['id']}"):
        for trainOrTest in ["train", "test"]:
            if trainOrTest not in args["trainOrTest"]:
                continue
            print(f"PID {os.getpid()} on step {i+1} of {trainOrTest}-data at {datetime.datetime.now()}")
            timeStamp = int(str(time()).replace('.', ''))
            rows = saveAllPosDifPatterns(trainOrTest, int(testDivider[trainOrTest]*20), timeStamp, BFDdiameter, processID=args["id"], silence=True, 
                                         structure = args["structure"], start=start, end=end, maxPooling=maxPooling, simple = True, 
                                         nonPredictedBorderInA=nonPredictedBorderInA, zernike=zernike, defocus=defocus, numberOfOSAANSIMoments= 20)
            #rows = saveAllDifPatterns(XDIMTILES, YDIMTILES, trainOrTest, int(12*testDivider[trainOrTest]), timeStamp, BFDdiameter, processID=args["id"], silence=True, maxPooling = maxPooling, structure = args["structure"], start=start, end=end)
            writeAllRows(rows=rows, trainOrTest=trainOrTest, XDIMTILES=XDIMTILES, YDIMTILES=YDIMTILES, processID=args["id"], 
                         timeStamp = timeStamp, maxPooling=maxPooling, size = size, zernike=zernike)
  
    print(f"PID {os.getpid()} done.")

       
