import glob
import os
import shutil
import struct
import time
from abtem.reconstruct import MultislicePtychographicOperator, RegularizedPtychographicOperator
import cv2
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from FullPixelGridML.SimulateTilesOneFile import generateDiffractionArray
import numpy as np
from ase.io import write
from ase.visualize.plot import plot_atoms
from abtem.measure import Measurement
from datasets import ptychographicDataLightning
from torch import nn
from lightningTrain import loadModel, lightnModelClass
from Zernike.ZernikeTransformer import Zernike, calc_diameter_bfd

DIMTILES = 12

def replace_line_in_file(file_path, line_number, text):

    """

    Replaces a specific line in a file with the given text.



    :param file_path: Path to the file where the line needs to be replaced.

    :param line_number: The line number (starting from 1) that needs to be replaced.

    :param text: The new text that will replace the existing line.

    """

    try:

        # Read the file contents

        with open(file_path, 'r') as file:

            lines = file.readlines()



        # Check if the specified line number is within the file's line count

        if line_number > len(lines) or line_number < 1:

            raise ValueError("Line number is out of range.")



        # Replace the specified line

        lines[line_number - 1] = text + '\n'



        # Write back the modified content

        with open(file_path, 'w') as file:

            file.writelines(lines)



    except FileNotFoundError:

        print(f"The file {file_path} was not found.")

    except IOError:

        print("An error occurred while reading or writing the file.")

    except Exception as e:

        print(f"An error occurred: {e}")

def cutEverythingBelowLine(textFile : str, line:int):
    """Get rid of everything below a certain line in a text file. Keeps the specified line.

    Args:
        textFile (str): _description_
        line (int): _description_
    """
    with open(textFile, 'r') as f:
        lines = f.readlines()
    with open(textFile, 'w') as f:
        for i in range(len(lines)):
            if i < line:
                f.write(lines[i])
            else:
                break

def writeParamsCNF(ScanX,ScanY, beamPositions, diameterBFD, conv_angle_in_mrad= 33, energy = 60e3, CBEDDim=50):
    probeDim = int(np.ceil(np.sqrt(2)*3*CBEDDim/2))
    theta_in_mrad = conv_angle_in_mrad
    h_times_c_dividedBy_keV_in_A = 12.4
    wavelength_in_A = h_times_c_dividedBy_keV_in_A/np.sqrt(2*511*energy/1000+(energy/1000)**2) #de Broglie wavelength
    reciprocalPixelSize =  2*(theta_in_mrad/1000)/(diameterBFD * wavelength_in_A ) #in 1/A
    realSpacePixelSize_in_A = 1/probeDim/reciprocalPixelSize #in Angstrom
    objectDim = 100*(int(probeDim/100) + 2)

    try:
        os.remove('Params.cnf')
        print("Old Params.cnf found. Replacing with new file.")
    except FileNotFoundError:
        print("Params.cnf not found. Creating new file.")
        pass
    shutil.copyfile('Params copy.cnf', 'Params.cnf') 
    cutEverythingBelowLine('Params.cnf', 65)
    with open('Params.cnf', 'a') as f:
        for i in range(len(beamPositions)):
            f.write(f"beam_position:   {beamPositions[i][0]:.4e} {beamPositions[i][1]:.4e}\n")
    conv_angle = conv_angle_in_mrad / 1000
    replace_line_in_file('Params.cnf', 5, f"E0: {energy:.0e}        #Acceleration voltage")
    replace_line_in_file('Params.cnf', 28, f"ObjAp:         {conv_angle}    #Aperture angle in Rad")
    replace_line_in_file('Params.cnf', 31, f"ObjectDim:  {objectDim}  #Object dimension")
    replace_line_in_file('Params.cnf', 32, f"ProbeDim:  {probeDim}  #Probe dimension")
    replace_line_in_file('Params.cnf', 33, f"PixelSize: {realSpacePixelSize_in_A:.6}e-10 #Real space pixel size")
    replace_line_in_file('Params.cnf', 37, f"CBEDDim:  {CBEDDim}  #CBED dimension")
    replace_line_in_file('Params.cnf', 44, f"ScanX:   {ScanX}     #Number of steps in X-direction")
    replace_line_in_file('Params.cnf', 45, f"ScanY:   {ScanY}     #Number of steps in Y-direction")
    replace_line_in_file('Params.cnf', 46, f"batchSize: {min(ScanX,ScanY)}  #Number of CBEDs that are processed in parallel")

energy = 60e3
conv_angle_in_mrad = 33

nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick = generateDiffractionArray(conv_angle= conv_angle_in_mrad, energy=energy,structure="MarcelsEx", pbar = True, start=[2, 0], end=[25, 15])
assert(measurement_thick.array.shape[2] == measurement_thick.array.shape[3])
CBEDDim = measurement_thick.array.shape[2]
measurementArray = np.zeros((measurement_thick.array.shape[0],measurement_thick.array.shape[1],CBEDDim,CBEDDim))

realPositions = np.zeros((measurementArray.shape[0],measurementArray.shape[1]), dtype = object)   
for i in range(measurementArray.shape[0]): 
    for j in range(measurementArray.shape[1]):
        realPositions[i,j] = (i*0.2*1e-10,j*0.2*1e-10)
for i in range(measurementArray.shape[0]):
    for j in range(measurementArray.shape[1]):
        measurementArray[i,j,:,:] = measurement_thick.array[i,j,:,:]

assert(measurementArray.shape[0] > DIMTILES)
assert(measurementArray.shape[1] > DIMTILES) 

        #measurementArray[i,j,:,:] = measurementArray[i,j,:,:]/(np.sum(measurementArray[i,j,:,:])+1e-10) 
diameterBFD = calc_diameter_bfd(measurementArray[0,0,:,:])
# leftEdge = CBEDDim
# rightEdge = 0
# for row in brightFieldDisk:
#     for i in range(len(row)):
#         if row[i] == 1 and i < leftEdge:
#             leftEdge = i
#         if i < len(row) - 1 and row[i+1] == 0 and row[i] == 1 and i > rightEdge:
#             rightEdge = i

# if rightEdge < leftEdge:
#     raise Exception("rightEdge < leftEdge")
    
# diameterBFD = rightEdge - leftEdge
print(f"diameter of the bfd in pixels: {diameterBFD}")

#remove ROP clutter
for filename in glob.glob('Probe*.bin'):
    os.remove(filename)
for filename in glob.glob('Potential*.bin'):
    os.remove(filename)
for filename in glob.glob('Positions*.txt'):
    os.remove(filename)
for filename in glob.glob('Potential*.png'):
    os.remove(filename)
        

writeParamsCNF(measurementArray.shape[1],measurementArray.shape[0], realPositions.flatten(), diameterBFD, conv_angle_in_mrad = conv_angle_in_mrad, energy=energy, CBEDDim=CBEDDim)
size = measurementArray.shape[0] * measurementArray.shape[1] * measurementArray.shape[2] * measurementArray.shape[3]
measurementArrayToFile = np.copy(measurementArray)
#Normalize data - required for ROP
measurementArrayToFile /= (np.sum(measurementArrayToFile)/(measurementArrayToFile.shape[0] * measurementArrayToFile.shape[1]))
#Convert to binary format
measurementArrayToFile = np.ravel(measurementArrayToFile)
measurementArrayToFile = struct.pack(size * 'f', *measurementArrayToFile)
file = open("testMeasurement.bin", 'wb')
file.write(measurementArrayToFile)
file.close()

# measurementArray.astype('float').flatten().tofile("testMeasurement.bin")











# write('testAtomStruct.png', atomStruct)
# print(len(potential_thick))

# randomizedXPositions = np.random.choice(measurement_thick.shape[0], int(min(measurement_thick.shape[0],measurement_thick.shape[1]) * 0.3), replace = True)
# randomizedYPositions = np.random.choice(measurement_thick.shape[1], int(min(measurement_thick.shape[0],measurement_thick.shape[1]) * 0.3), replace = True)

# realPositionsSparse = np.ravel(realPositions[randomizedXPositions, randomizedYPositions])

# measurementArraySparse = np.copy(measurement_thick)
# measurementArraySparse = np.ravel(measurement_thick[randomizedXPositions, randomizedYPositions,:,:])



multislice_reconstruction_ptycho_operator = RegularizedPtychographicOperator(
    measurement_thick,
    scan_step_sizes = 0.2,
    semiangle_cutoff=conv_angle_in_mrad,
    energy=energy,
    #num_slices=10,
    device="gpu",
    #slice_thicknesses=1
).preprocess()

mspie_objects, mspie_probes, rpie_positions, mspie_sse = multislice_reconstruction_ptycho_operator.reconstruct(
    max_iterations=20, return_iterations=True, random_seed=1, verbose=True
)

mspie_objects[-1].angle().interpolate(potential_thick.sampling).show()

print("Calculating the number of Zernike moments:")
resultVectorLength = 0 
numberOfOSAANSIMoments = 20	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1

print("load model")
model = loadModel(modelName = "ZernikeNormal", numChannels=resultVectorLength * DIMTILES**2, numLabels = (DIMTILES**2)//9)

model = lightnModelClass.load_from_checkpoint(checkpoint_path = os.path.join("checkpoints",f"ZernikeNormal_PixelReworkedZernikeAndNewDatasets","epoch=23-step=162840.ckpt"), model = model)
model.eval()

#initiate Zernike
radius = diameterBFD//2 + 1
ZernikeObject = Zernike(radius, numberOfOSAANSIMoments= numberOfOSAANSIMoments)

# def generateGroupOfPatterns(measurementArray):
#     for i in range((measurementArray.shape[0]-DIMTILES)):
#         for j in range((measurementArray.shape[1]-DIMTILES)):
#             difPatternsOnePosition = np.copy(measurementArray[i:DIMTILES+i,j:DIMTILES+j,:,:])
#             yield np.reshape(difPatternsOnePosition, (-1,difPatternsOnePosition.shape[-2], difPatternsOnePosition.shape[-1]))

mASomeZeroedOut = np.copy(measurementArray)
mASomeZeroedOut = np.reshape(mASomeZeroedOut, (-1,mASomeZeroedOut.shape[2],mASomeZeroedOut.shape[3]))
mASomeZeroedOut[np.random.choice(mASomeZeroedOut.shape[0], int(0*mASomeZeroedOut.shape[0]), replace = False),:,:] = np.zeros((mASomeZeroedOut.shape[-2],mASomeZeroedOut.shape[-1]))
mASomeZeroedOut = np.reshape(mASomeZeroedOut, (measurementArray.shape[0],measurementArray.shape[1],measurementArray.shape[2],measurementArray.shape[3]))

#ZernikeTransform
print("generating the zernike moments")

poolingFactor = int(3)

# loop over the all positions and apply the model to the data
Predictions = np.zeros(measurementArray.shape[:2])
for i in tqdm(range(measurementArray.shape[0]-DIMTILES), desc= "Going through all positions and predicting"):
    for j in range(measurementArray.shape[1]-DIMTILES):
        groupOfPatterns = mASomeZeroedOut[i:DIMTILES+i,j:DIMTILES+j,:,:]
        groupOfPatterns = np.reshape(groupOfPatterns, (-1,groupOfPatterns.shape[-2], groupOfPatterns.shape[-1]))
        timeStart = time.time()
        zernikeValues = ZernikeObject.zernikeTransform(fileName = None, groupOfPatterns = groupOfPatterns, zernikeTotalImages = None)
        tqdm.write(f"Time elapsed to zernike transform: {timeStart-time.time()}")
        pred = model(torch.tensor(zernikeValues).float())
        pred = pred.detach().numpy()
        # xMostLikely = pred[3]//0.2 + i
        # yMostLikely = pred[6]//0.2 + j
        # if xMostLikely < 0 or xMostLikely >= Predictions.shape[0] or yMostLikely < 0 or yMostLikely >= Predictions.shape[1]:
        #     continue
        # Predictions[int(xMostLikely),int(yMostLikely)] += 1
        predExpanded = np.zeros((DIMTILES, DIMTILES))
        for n in range(poolingFactor):
            predExpanded[n::poolingFactor,n::poolingFactor] = pred.reshape((DIMTILES//poolingFactor,DIMTILES//poolingFactor))
        try:
            Predictions[i:min(i+DIMTILES, Predictions.shape[0]),j:min(j+DIMTILES, Predictions.shape[1])] += predExpanded[:min(DIMTILES, Predictions.shape[0]-i),:min(DIMTILES, Predictions.shape[1]-j)]
        except Exception as e:
            print(f"Ignored Error: {e}")

# for all elements in the predictions array divide through coordinates
for x in range(Predictions.shape[0]):
    for y in range(Predictions.shape[1]):
        xArea = min(DIMTILES, x+1, Predictions.shape[0]-x)
        yArea = min(DIMTILES, y+1, Predictions.shape[1]-y)
        Predictions[x,y] = Predictions[x,y]/(xArea*yArea)

Predictions/=np.max(Predictions)


plt.tight_layout()
plt.savefig("testStructureFullRec.png")
plt.imshow(Predictions)
plt.colorbar()
plt.imsave("Predictions.png", Predictions)
plt.savefig("testStructureFullRec_WithPredictions.png")
plt.close()
plt.imsave("PredictionsLog.png", np.log(Predictions+1))

