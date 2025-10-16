import os
import sys
from tqdm import tqdm

from Reconstruction import full_routine, parse_args_full


allstructures = []
#go through all files in the folder
for file in os.listdir("/data/scratch/haertelk/Masterarbeit/FullPixelGridML/structures"):
    if file.endswith(".cif"):
        allstructures.append("/data/scratch/haertelk/Masterarbeit/FullPixelGridML/structures/" + file)


structure_int, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates = parse_args_full()
temp_save  = [structure_int, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates]
print(f"Structure Int: {structure_int}")
for cnt, structure in enumerate(allstructures):  
    if cnt % 10 != int(structure_int):
        continue
    print(f"Processing structure {cnt + 1}/{len(allstructures)}: {structure}")
    # Call the full routine with the specified parameters
    try:
        full_routine(structure, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates)
    except ValueError as e:
        print(f"Error processing {structure}: {e}")
        continue
    structure_int, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates = temp_save
print("Fertig")
