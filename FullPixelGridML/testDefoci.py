import numpy as np
from SimulateTilesOneFile import generateDiffractionArray
from tqdm import tqdm

energy = 60e3
for defocus in tqdm([0, -10, -30, -50, -100, -150]):
    generateDiffractionArray(trainOrTest = None, conv_angle = 33, energy = energy, 
                                structure = "/data/scratch/haertelk/Masterarbeit/FullPixelGridML/structures/used/NaSbF6.cif", pbar = False, 
                                start = (5,5), end = (20,20), simple = True,
                                generate_graphics = True, defocus = defocus)
    
h_times_c_dividedBy_keV_in_A = 12.4
wavelength_in_A = h_times_c_dividedBy_keV_in_A/np.sqrt(2*511*energy/1000+(energy/1000)**2)
print(f"wavelength_in_A: {wavelength_in_A}")
print(f"airy disk diameter first order: {1.22 * wavelength_in_A / 0.033}")