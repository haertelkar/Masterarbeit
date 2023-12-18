from abtem.reconstruct import MultislicePtychographicOperator, RegularizedPtychographicOperator
from matplotlib import pyplot as plt
from FullPixelGridML.SimulateTilesOneFile import generateDiffractionArray
import numpy as np
from ase.io import write
from ase.visualize.plot import plot_atoms
from abtem.measure import Measurement

nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick = generateDiffractionArray()

write('testAtomStruct.png', atomStruct)
print(len(potential_thick))

measurementArray = np.copy(measurement_thick)
measurementArray[np.random.choice(measurementArray.shape[0], 20), np.random.choice(measurementArray.shape[0], 20), :, :] = np.zeros_like(measurementArray[0,0])

multislice_reconstruction_ptycho_operator = RegularizedPtychographicOperator(
    measurement_thick,
    scan_step_sizes = 0.2,
    semiangle_cutoff=24,
    energy=200e3,
    #num_slices=10,
    device="gpu",
    #slice_thicknesses=0.2
).preprocess()

mspie_objects, mspie_probes, rpie_positions, mspie_sse = multislice_reconstruction_ptycho_operator.reconstruct(
    max_iterations=5, return_iterations=True, random_seed=1, verbose=True
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

mspie_objects[-1].angle().interpolate(potential_thick.sampling).show(ax=ax2) #[-1][0]

fig.tight_layout()

plt.savefig("test.pdf")
plt.close()

m = Measurement(measurementArray, measurement_thick.calibrations, measurement_thick.units)

multislice_reconstruction_ptycho_operator = RegularizedPtychographicOperator(
    m,
    scan_step_sizes = 0.2,
    semiangle_cutoff=24,
    energy=200e3,
    #num_slices=10,
    #device="gpu",
    #slice_thicknesses=0.2
).preprocess()

mspie_objects, mspie_probes, rpie_positions, mspie_sse = multislice_reconstruction_ptycho_operator.reconstruct(
    max_iterations=40, return_iterations=True, random_seed=1, verbose=True
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

mspie_objects[-1].angle().interpolate(potential_thick.sampling).show(ax=ax2) #[-1][0]

fig.tight_layout()

plt.savefig("sparseTest.pdf")