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
from FullPixelGridML.SimulateTilesOneFile import generateDiffractionArray, createAllXYCoordinates, saveAllPosDifPatterns
import numpy as np
from ase.io import write
from ase.visualize.plot import plot_atoms
from abtem.measure import Measurement
from datasets import ptychographicDataLightning
from torch import nn
from lightningTrain import loadModel, lightnModelClass, DQNLightning
from Zernike.ZernikeTransformer import Zernike, calc_diameter_bfd



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



def createMeasurementArray(DIMTILES, energy, conv_angle_in_mrad, structure):
    nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick = generateDiffractionArray(conv_angle= conv_angle_in_mrad, energy=energy,structure=structure, pbar = True, start=[5, 5], end=[8, 8])
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
    return nameStruct,gridSampling,atomStruct,measurement_thick,potential_thick,CBEDDim,measurementArray,realPositions



        #measurementArray[i,j,:,:] = measurementArray[i,j,:,:]/(np.sum(measurementArray[i,j,:,:])+1e-10) 



def CleanUpROP():
    #remove ROP clutter
    for filename in glob.glob('Probe*.bin'):
        os.remove(filename)
    for filename in glob.glob('Potential*.bin'):
        os.remove(filename)
    for filename in glob.glob('Positions*.txt'):
        os.remove(filename)
    for filename in glob.glob('Potential*.png'):
        os.remove(filename)

def createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArray, realPositions, diameterBFD):
    CleanUpROP()
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

def plotGTandPred(atomStruct, groundTruth, Predictions):
    """Create plots of the ground truth and the predictions. 
    They are transposed because the reconstruction is also transposed.
    """
    extent=  [5,8,5,8]
    plt.imshow(Predictions.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig("Predictions.png")
    plt.close()
    plt.imshow(np.log(Predictions+1).T, origin = "lower", extent=extent)
    plt.savefig("PredictionsLog.png")
    plt.close()
    import abtem
    abtem.show_atoms(atomStruct, legend = True)
    plt.savefig("testStructureOriginal.png")
    plt.close()
    plt.imshow(groundTruth.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig("groundTruth.png")
    plt.close()
    plt.imshow(np.log(groundTruth+1).T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig("groundTruthLog.png")
    plt.close()
    print("Finished")

def createAndPlotReconstructedPotential(potential_thick):
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
    max_iterations=20, return_iterations=True, random_seed=1, verbose=True)

    mspie_objects[-1].angle().interpolate(potential_thick.sampling).show(extent = [5,20,5,20])
    plt.tight_layout()
    plt.savefig("testStructureFullRec.png")

def groundTruthCalculator(DIMTILES, LabelCSV, groundTruth, Predictions):
    for i in range(10):
        gtDist = [LabelCSV[i*3], LabelCSV[1 + i*3]]
        xGT = int(np.around(float(gtDist[0])))
        yGT = int(np.around(float(gtDist[1])))
        element = int(float(LabelCSV[2+i*3]))
        print(f"atom no. {i}: gt element: {element}, Ground truth position: {xGT}, {yGT}, Ground truth values: {gtDist}")
        if xGT < 0 or xGT >= Predictions.shape[0] or yGT < 0 or yGT >= Predictions.shape[1]:
            print("Ground truth out of bounds")
        else:
            groundTruth[int(xGT), int(yGT)] += 1#element


# measurementArray.astype('float').flatten().tofile("testMeasurement.bin")
energy = 60e3
structure="multiPillars"
conv_angle_in_mrad = 33
DIMTILES = 10
DIMENSION = DIMTILES//3 + 1
dim = 50
diameterBFD50Pixels = 18
print("Calculating the number of Zernike moments:")
resultVectorLength = 0 
numberOfOSAANSIMoments = 15	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1



nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick, CBEDDim, measurementArray, realPositions = createMeasurementArray(DIMTILES, energy, conv_angle_in_mrad, structure) 
diameterBFDNotScaled = calc_diameter_bfd(measurementArray[0,0,:,:])
plt.imsave("detectorImage.png",measurementArray[0,0,:,:])
createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArray, realPositions, diameterBFDNotScaled)
print(f"Calculated BFD not scaled to 50 pixels: {diameterBFDNotScaled}")
print(f"imageDim Not Scaled to 50 Pixels: {measurementArray[0,0,:,:].shape[-1]}")
print(f"Calculated BFD scaled to 50 pixels: {diameterBFDNotScaled/measurementArray[0,0,:,:].shape[-1] * 50}")
print(f"Actually used BFD with margin: {diameterBFD50Pixels}")



# createAndPlotReconstructedPotential(potential_thick)

#model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)

model = DQNLightning().load_from_checkpoint(checkpoint_path = os.path.join("checkpoints/DQN_0711_chamfer_1e-9_BatS4096_AdamW_1343/epoch=331-step=996.ckpt"))
model.eval()

#initiate Zernike
print("generating the zernike moments")
xMaxCNT, yMaxCNT = np.shape(measurement_thick.array)[:2] # type: ignore


#ZernikeTransform
difArrays = [[nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick]]

RelLabelCSV, dataArray = saveAllPosDifPatterns(None, -1, None, diameterBFD50Pixels, processID = 99999, silence = False, maxPooling = 1, structure = "None", fileWrite = False, difArrays = difArrays, start = (5,5), end = (8,8)) # type: ignore
RelLabelCSV, dataArray = RelLabelCSV[0][1:], dataArray[0]
print(f"RelLabelCSV : {RelLabelCSV}")
radius = dataArray.shape[-1] #already removed everything outside BFD in saveAllDifPatterns 
ZernikeObject = Zernike(radius, numberOfOSAANSIMoments= numberOfOSAANSIMoments)
#poolingFactor = int(3)
groundTruth = np.zeros((measurementArray.shape[0], measurementArray.shape[1]))
# loop over the all positions and apply the model to the data
Predictions = np.zeros_like(groundTruth)

# predictions = []

# for cnt, (xSteps, ySteps) in tqdm(enumerate(createAllXYCoordinates(yMaxStartCoord,xMaxStartCoord))   , desc= "Going through all positions and predicting"):
#     zernikeValues = ZernikeObject.zernikeTransform(fileName = None, groupOfPatterns = dataArray[cnt], zernikeTotalImages = None)
#     pred = model(torch.tensor(zernikeValues).float())
#     pred = pred.detach().numpy()
    
#     xCNT = xStepSize * xSteps
#     yCNT = yStepSize * ySteps
#     predictions.append(pred)

# predictionArray = np.array(predictions)
# medianPredicitionX = np.median(predictionArray[:,0])
# print(f"Median prediction for x: {medianPredicitionX}")
# medianPredicitionY = np.median(predictionArray[:,1])
# print(f"Median prediction for y: {medianPredicitionY}")


# for cnt, (xSteps, ySteps) in tqdm(enumerate(createAllXYCoordinates(yMaxStartCoord,xMaxStartCoord))   , desc= "Going through all positions and predicting"):

zernikeValues = ZernikeObject.zernikeTransform(fileName = None, groupOfPatterns = dataArray, zernikeTotalImages = None)
zernikeValues = torch.tensor(zernikeValues).float()
# randCoords = torch.randperm(225)[:9]
# randXCoords = (randCoords % 15)
# randYCoords = torch.div(randCoords, 15, rounding_mode='floor') 
randXCoords = torch.tensor([0, 0, 0, 7, 7, 7, 14, 14, 14])
randYCoords = torch.tensor([0, 7, 14, 0, 7, 14, 0, 7, 14])
imageOrZernikeMoments = zernikeValues.reshape((15,15, -1))[randXCoords,randYCoords].reshape((9,-1))
# imageOrZernikeMomentsWithCoords = torch.cat((imageOrZernikeMoments, torch.stack([randXCoords, randYCoords]).T), dim = 1)
with torch.inference_mode():
    pred = model(imageOrZernikeMoments.unsqueeze(0))
    pred = pred.detach().numpy().reshape((10,2))

for predXY in pred:
    xMostLikely = np.clip(np.round(predXY[0]),0,14).astype(int)
    yMostLikely = np.clip(np.round(predXY[1]),0,14).astype(int)

    print(f"Predicted position: {xMostLikely}, {yMostLikely}, Predicted values: {predXY}")
    if xMostLikely < 0 or xMostLikely >= Predictions.shape[0] or yMostLikely < 0 or yMostLikely >= Predictions.shape[1]:
        print("Prediction out of bounds")
    else:
        Predictions[xMostLikely, yMostLikely] += 1 #for some reason the reconstruction is transposed. Is adjusted later when plotted

groundTruthCalculator(DIMTILES, RelLabelCSV, groundTruth, Predictions)
        # groundTruth[xCNT + DIMTILES//2, yCNT + DIMTILES//2] += 1 only for plotting the tile centers
    # predExpanded = np.zeros((DIMTILES, DIMTILES))
    # for n in range(poolingFactor):
    #     predExpanded[n::poolingFactor,n::poolingFactor] = pred.reshape((DIMTILES//poolingFactor,DIMTILES//poolingFactor))
    # try:
    #     Predictions[i:min(i+DIMTILES, Predictions.shape[0]),j:min(j+DIMTILES, Predictions.shape[1])] += predExpanded[:min(DIMTILES, Predictions.shape[0]-i),:min(DIMTILES, Predictions.shape[1]-j)]
    # except Exception as e:
    #     print(f"Ignored Error: {e}")

# for all elements in the predictions array divide through coordinates
# for x in range(Predictions.shape[0]):
#     for y in range(Predictions.shape[1]):
#         xArea = min(DIMTILES, x+1, Predictions.shape[0]-x)
#         yArea = min(DIMTILES, y+1, Predictions.shape[1]-y)
#         Predictions[x,y] = Predictions[x,y]/(xArea*yArea)





plotGTandPred(atomStruct, groundTruth, Predictions)
