from ase import Atoms
from ase.visualize import view
from ase.build import surface
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
import torch
from tqdm import tqdm
import csv
import warnings

kindsOfElements = {6:0, 14:1, 74:2}
device = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Calculating on {device}")

for trainOrTest in ["train", "test"]:
    with open('measurements_{trainOrTest}\\labels.csv'.format(trainOrTest = trainOrTest), 'w+', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        Writer.writerow(["fileName", "element", "xAtomRel", "xAtomShift", "yAtomRel", "yAtomShift", "zAtoms"])
        Writer = None
    for i in tqdm(range(20)):
        xlen = 5
        ylen = 5
        element = choice([6,14,74])

        xAtom = random()*xlen/3 + xlen/3  #x position of first atom (in inner third)
        yAtom = random()*ylen/3 + ylen/3 #y position of first atom (in inner third)
        zAtoms = randint(1,10) #number of atoms in z axis
        maxShiftFactorPerLayer = 0.01
        xAtomShift = 0#(random()  - 1/2) * maxShiftFactorPerLayer * xlen  #slant of atom pillar
        yAtomShift = 0#(random() -  1/2) * maxShiftFactorPerLayer * ylen #slant of atom pillar
        xAtomPillarAngle = np.arcsin(xAtomShift/1)
        yAtomPillarAngle = np.arcsin(yAtomShift/1)
        positions = [np.array([xAtom, yAtom, 0])]
        for atom in range(zAtoms-1):
            positions += [positions[atom] + np.array([xAtomShift, yAtomShift, 1])]
        atomPillar = Atoms(numbers = [element] * zAtoms , positions=positions, cell = [xlen, ylen, zAtoms,90,90,90])
        # view(atomPillar)
        # plt.show()

        potential_thick = Potential(
            atomPillar,
            sampling=0.02,
            parametrization="kirkland",
            device=device
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = Probe(semiangle_cutoff=24, energy=200e3, device=device)
            probe.match_grid(potential_thick)

            pixelated_detector = PixelatedDetector(max_angle=120)
            gridscan = GridScan(
                (0, 0), potential_thick.extent, sampling=0.2
            )
            measurement_thick = probe.scan(gridscan, pixelated_detector, potential_thick, pbar = False)
        element = kindsOfElements[element]
        with open('measurements_{trainOrTest}\\labels.csv'.format(trainOrTest = trainOrTest), 'a', newline='') as csvfile:
            Writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for xCNT, difPatternRow in enumerate(measurement_thick.array):
                for yCNT, difPattern in enumerate(difPatternRow):
                    if yCNT%2 or xCNT%2:
                        continue #less data from a single positio         
                    xPos = xCNT * 0.2
                    yPos = yCNT * 0.2
                    xAtomRel = xAtom - xPos
                    yAtomRel = yAtom - yPos
                    fileName = "measurements_{trainOrTest}\\{element}_{xAtomRel}_{xAtomShift}_{yAtomRel}_{yAtomShift}_{zAtoms}.npy"
                    fileName = fileName.format(trainOrTest = trainOrTest, element = element, xAtomRel = xAtomRel, xAtomShift = xAtomShift, yAtomRel = yAtomRel, yAtomShift = yAtomShift, zAtoms = zAtoms)
                    np.save(fileName,difPattern)
                    Writer.writerow([fileName.split("\\")[-1]] + [str(difParams) for difParams in [element, xAtomRel, xAtomShift, yAtomRel, yAtomShift, zAtoms]])
            
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