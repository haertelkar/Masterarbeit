import glob
import os
import shutil
import struct
import sys
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
from lightningTrain import loadModel, lightnModelClass, TwoPartLightning
from Zernike.ZernikeTransformer import Zernike, calc_diameter_bfd
from pympler import asizeof



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

def writeParamsCNF(ScanX,ScanY, beamPositions, diameterBFD, conv_angle_in_mrad= 33, energy = 60e3, CBEDDim=50, folder = ".", grid = False):
    probeDim = int(np.ceil(np.sqrt(2)*3*CBEDDim/2))
    theta_in_mrad = conv_angle_in_mrad
    h_times_c_dividedBy_keV_in_A = 12.4
    wavelength_in_A = h_times_c_dividedBy_keV_in_A/np.sqrt(2*511*energy/1000+(energy/1000)**2) #de Broglie wavelength
    reciprocalPixelSize =  2*(theta_in_mrad/1000)/(diameterBFD * wavelength_in_A ) #in 1/A
    realSpacePixelSize_in_A = 1/probeDim/reciprocalPixelSize #in Angstrom
    objectDim = 100*(int(probeDim/100) + 2)

    try:
        os.remove(f'{folder}/Params.cnf')
        print("Old Params.cnf found. Replacing with new file.")
    except FileNotFoundError:
        print(f"{folder}/Params.cnf not found. Creating new file.")
        pass
    shutil.copyfile('Params copy.cnf', f'{folder}/Params.cnf') 
    cutEverythingBelowLine(f'{folder}/Params.cnf', 65)
    with open(f'{folder}/Params.cnf', 'a') as f:
        for beamPosition in beamPositions:
            f.write(f"beam_position:   {beamPosition[0]:.4e} {beamPosition[1]:.4e}\n")
    conv_angle = conv_angle_in_mrad / 1000
    replace_line_in_file(f'{folder}/Params.cnf', 5, f"E0: {energy:.0e}        #Acceleration voltage")
    replace_line_in_file(f'{folder}/Params.cnf', 28, f"ObjAp:         {conv_angle}    #Aperture angle in Rad")
    replace_line_in_file(f'{folder}/Params.cnf', 31, f"ObjectDim:  {objectDim}  #Object dimension")
    replace_line_in_file(f'{folder}/Params.cnf', 32, f"ProbeDim:  {probeDim}  #Probe dimension")
    replace_line_in_file(f'{folder}/Params.cnf', 33, f"PixelSize: {realSpacePixelSize_in_A:.6}e-10 #Real space pixel size")
    replace_line_in_file(f'{folder}/Params.cnf', 37, f"CBEDDim:  {CBEDDim}  #CBED dimension")
    replace_line_in_file(f'{folder}/Params.cnf', 40, f"ptyMode: {int(not grid)}      #Input positions: 0 == grid scan; 1 == arbitrary (need to be specified below)")
    replace_line_in_file(f'{folder}/Params.cnf', 44, f"ScanX:   {ScanX}     #Number of steps in X-direction")
    replace_line_in_file(f'{folder}/Params.cnf', 45, f"ScanY:   {ScanY}     #Number of steps in Y-direction")
    replace_line_in_file(f'{folder}/Params.cnf', 46, f"batchSize: {min(ScanX,ScanY)}  #Number of CBEDs that are processed in parallel")



def createMeasurementArray(energy, conv_angle_in_mrad, structure, start, end, DIMTILES = 15):
    nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick = generateDiffractionArray(conv_angle= conv_angle_in_mrad, energy=energy,structure=structure, pbar = True, start=start, end=end)
    assert(measurement_thick.array.shape[2] == measurement_thick.array.shape[3])
    print(f"nameStruct: {nameStruct}")
    CBEDDim = measurement_thick.array.shape[2]
    measurementArray = np.zeros((measurement_thick.array.shape[0],measurement_thick.array.shape[1],CBEDDim,CBEDDim))

    realPositions = np.zeros((measurementArray.shape[0],measurementArray.shape[1]), dtype = object)   
    for i in range(measurementArray.shape[0]): 
        for j in range(measurementArray.shape[1]):
            realPositions[i,j] = ((i-measurementArray.shape[0]/2)*0.2*1e-10,(j-measurementArray.shape[1]/2)*0.2*1e-10)
    for i in range(measurementArray.shape[0]):
        for j in range(measurementArray.shape[1]):
            measurementArray[i,j,:,:] = measurement_thick.array[i,j,:,:]

    assert(measurementArray.shape[0] > DIMTILES)
    assert(measurementArray.shape[1] > DIMTILES)
    return nameStruct,gridSampling,atomStruct,measurement_thick,potential_thick,CBEDDim,measurementArray, realPositions.flatten()



        #measurementArray[i,j,:,:] = measurementArray[i,j,:,:]/(np.sum(measurementArray[i,j,:,:])+1e-10) 



def CleanUpROP():
    #remove ROP clutter
    for folder in [".","PredROP","TotalPosROP","TotalGridROP"]:
        for filename in glob.glob('Probe*.bin'):
            os.remove(f"{folder}/{filename}")
        for filename in glob.glob('Potential*.bin'):
            os.remove(f"{folder}/{filename}")
        for filename in glob.glob('Positions*.txt'):
            os.remove(f"{folder}/{filename}")
        for filename in glob.glob('Potential*.png'):
            os.remove(f"{folder}/{filename}")

def createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile, allKnownPositionsInA, diameterBFD, fullLengthX, fullLengthY, folder = ".", grid = False):
    CleanUpROP()
    writeParamsCNF(fullLengthX,fullLengthY, allKnownPositionsInA, diameterBFD, conv_angle_in_mrad = conv_angle_in_mrad, energy=energy, CBEDDim=CBEDDim, folder = folder, grid = grid)
    size = np.prod(measurementArrayToFile.shape)
    #Normalize data - required for ROP - now done before
    # measurementArrayToFile /= (np.sum(measurementArrayToFile)/(measurementArrayToFile.shape[0] * measurementArrayToFile.shape[1]))
    # measurementArrayToFile /= np.sum(measurementArrayToFile)/len(allKnownPositionsMeasurements)
    #Convert to binary format
    for name in dir():
        try:
            if not name.startswith('_'):
                print(f"Name: {name}")
                print(f"\tsize: {asizeof.asizeof(eval(name))}")
        except:
            pass
    
    chunk_size = 1024 * 1024  # Number of floats per chunk, adjust as needed
    num_elements = len(measurementArrayToFile)  # Total number of elements
    with open(f"{folder}/testMeasurement.bin", 'wb') as file:
        for i in range(0, num_elements, chunk_size):
            # Determine the chunk size dynamically to handle the last chunk
            current_chunk_size = min(chunk_size, num_elements - i)
            chunk = measurementArrayToFile[i:i + current_chunk_size]
            packed_chunk = struct.pack(current_chunk_size * 'f', *chunk)
            file.write(packed_chunk)

    # measurementArrayToFile = struct.pack(size * 'f', *measurementArrayToFile)
    # file = open(f"{folder}/testMeasurement.bin", 'wb')
    # file.write(measurementArrayToFile)
    # file.close()

def plotGTandPred(atomStruct, groundTruth, Predictions, start, end, allKnownPositions):
    """Create plots of the ground truth and the predictions. 
    They are transposed because the reconstruction is also transposed.
    """
    extent=  [start[0],end[0],start[1],end[1]]
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
    GTinAllKnown = np.zeros_like(groundTruth)
    for pos in allKnownPositions:
        GTinAllKnown[pos] = groundTruth[pos]
    plt.imshow(GTinAllKnown.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig("GTTimesPred.png")
    plt.close()
    print("Finished")

def createAndPlotReconstructedPotential(potential_thick, start, end):
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

    mspie_objects[-1].angle().interpolate(potential_thick.sampling).show(extent = [start[0],end[0],start[1],end[1]])
    plt.tight_layout()
    plt.savefig("testStructureFullRec.png")

def groundTruthCalculator(LabelCSV, groundTruth):
    for i in range(len(LabelCSV)//3):
        gtDist = [LabelCSV[i*3], LabelCSV[1 + i*3]]
        xGT = int(np.around(float(gtDist[0])))
        yGT = int(np.around(float(gtDist[1])))
        element = int(float(LabelCSV[2+i*3]))
        print(f"atom no. {i}: gt element: {element}, Ground truth position: {xGT}, {yGT}, Ground truth values: {gtDist}")
        if xGT < 0 or xGT >= groundTruth.shape[0] or yGT < 0 or yGT >= groundTruth.shape[1]:
            print("Ground truth out of bounds")
        else:
            groundTruth[int(xGT), int(yGT)] += 1#element


# measurementArray.astype('float').flatten().tofile("testMeasurement.bin")
energy = 60e3
structure="random"
conv_angle_in_mrad = 33
dim = 50
diameterBFD50Pixels = 18
start = (5,5)
end = (25,25)

print("Calculating the number of Zernike moments:")
resultVectorLength = 0 
numberOfOSAANSIMoments = 15	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1



nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick, CBEDDim, measurementArray, realPositions = createMeasurementArray(energy, conv_angle_in_mrad, structure, start, end) 
diameterBFDNotScaled = calc_diameter_bfd(measurementArray[0,0,:,:])
plt.imsave("detectorImage.png",measurementArray[0,0,:,:])

print(f"Calculated BFD not scaled to 50 pixels: {diameterBFDNotScaled}")
print(f"imageDim Not Scaled to 50 Pixels: {measurementArray[0,0,:,:].shape[-1]}")
print(f"Calculated BFD scaled to 50 pixels: {diameterBFDNotScaled/measurementArray[0,0,:,:].shape[-1] * 50}")
print(f"Actually used BFD with margin: {diameterBFD50Pixels}")



# createAndPlotReconstructedPotential(potential_thick)

#model = loadModel(modelName = modelName, numChannels=numChannels, numLabels = numLabels)

model = TwoPartLightning().load_from_checkpoint(checkpoint_path = os.path.join("checkpoints/DQN_1312_1044_Z_GRU_onlyLastHid_30RandPosWithCoords_9000E/epoch=1609-step=24150.ckpt"))
model.eval()


xMaxCNT, yMaxCNT = np.shape(measurement_thick.array)[:2] # type: ignore


#ZernikeTransform
difArrays = [[nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick]]
del measurement_thick, potential_thick

RelLabelCSV, dataArray = saveAllPosDifPatterns(None, -1, None, diameterBFD50Pixels, processID = 99999, silence = False, maxPooling = 1, structure = "None", fileWrite = False, difArrays = difArrays, start = (5,5), end = (20,20)) # type: ignore
del difArrays
RelLabelCSV, dataArray = RelLabelCSV[0][1:], dataArray[0]
print(f"RelLabelCSV : {RelLabelCSV}")
radius = dataArray.shape[-1]//2 #already removed everything outside BFD in saveAllDifPatterns 
ZernikeObject = Zernike(radius, numberOfOSAANSIMoments= numberOfOSAANSIMoments)
#poolingFactor = int(3)
groundTruth = np.zeros((measurementArray.shape[0], measurementArray.shape[1]))


plt.imsave("detectorImage.png",dataArray.reshape(int(end[0]-start[1])*5,int(end[0]-start[1])*5,20,20)[0,0])

circleWeights = np.array([
    [1, 1, 1, 1, 1],
    [1, 3, 3, 3, 1],
    [1, 3, 2, 3, 1],
    [1, 3, 3, 3, 1],
    [1, 1, 1, 1, 1]
])


#initiate Zernike
print("generating the zernike moments")
zernikeValuesTotal = ZernikeObject.zernikeTransform(fileName = None, groupOfPatterns = dataArray, zernikeTotalImages = None)
del dataArray
zernikeValuesTotal = torch.tensor(zernikeValuesTotal).float().reshape(int(end[0]-start[1])*5,int(end[0]-start[1])*5, -1)

# loop over the all positions and apply the model to the data
allKnownPositions = []
Predictions = np.zeros_like(groundTruth)
PositionsToScanXSingle = np.arange(0, Predictions.shape[0], 3)
PositionsToScanYSingle = np.arange(0, Predictions.shape[1], 3)
PositionsToScanX = np.sort(np.array([PositionsToScanXSingle for i in range(len(PositionsToScanYSingle))]).flatten())
PositionsToScanY= np.array([PositionsToScanYSingle for i in range(len(PositionsToScanXSingle))]).flatten()

for indexX in tqdm(range(Predictions.shape[0]-15), desc  = "Going through all positions and predicting"):
    for indexY in range(Predictions.shape[1]-15):
        zernikeValues = zernikeValuesTotal[indexX :indexX  + 15, indexY :indexY + 15, :].reshape(225, resultVectorLength)
        zernikeValues = torch.tensor(zernikeValues).float()
        # numberOfPositions = 9
        # randCoords = torch.randperm(225)[:numberOfPositions]
        # XCoords = (randCoords % 15)
        # YCoords = torch.div(randCoords, 15, rounding_mode='floor') 
        # XCoordsLocal = torch.tensor([0, 0, 0, 7, 7, 7, 14, 14, 14])
        # YCoordsLocal = torch.tensor([0, 7, 14, 0, 7, 14, 0, 7, 14])
        # only extract values less than 15 from PositionsToScanX and PositionsToScanY
        PositionsToScanXLocal = PositionsToScanX - indexX
        PositionsToScanYLocal = PositionsToScanY - indexY
        mask = (PositionsToScanXLocal >= 0) * (PositionsToScanYLocal >= 0) * (PositionsToScanXLocal < 15) * (PositionsToScanYLocal < 15)
        PositionsToScanXLocal = PositionsToScanXLocal[mask]
        PositionsToScanYLocal = PositionsToScanYLocal[mask]
        YCoordsLocal = torch.tensor(PositionsToScanYLocal)
        XCoordsLocal = torch.tensor(PositionsToScanXLocal)
        # print(f"XCoordsLocal: {XCoordsLocal}")
        # print(f"YCoordsLocal: {YCoordsLocal}")
        

        imageOrZernikeMoments = zernikeValues.reshape((15,15,-1))[XCoordsLocal,YCoordsLocal].reshape((XCoordsLocal.shape[0],-1))
        imageOrZernikeMoments = torch.cat((imageOrZernikeMoments, torch.stack([XCoordsLocal, YCoordsLocal]).T), dim = 1)
        # imageOrZernikeMomentsWithCoords = torch.cat((imageOrZernikeMoments, torch.stack([randXCoords, randYCoords]).T), dim = 1)
        with torch.inference_mode():
            pred = model(imageOrZernikeMoments.unsqueeze(0))
            pred = pred.detach().numpy().reshape((1,10,2))[0]

        for predXY in pred:
            for xCircle in range(-2,3):
                for yCircle in range(-2,3):                  
                    xMostLikely = np.clip(np.round(predXY[0]),0,14).astype(int)+xCircle
                    yMostLikely = np.clip(np.round(predXY[1]),0,14).astype(int)+yCircle

                    # print(f"Predicted position: {xMostLikely + indexX}, {yMostLikely + indexY}, Predicted values: {predXY}")
                    if xMostLikely + indexX < 0 or xMostLikely + indexX>= Predictions.shape[0] or yMostLikely + indexY< 0 or yMostLikely+ indexY >= Predictions.shape[1]:
                        if xCircle == 0 and yCircle == 0:
                            tqdm.write("Predicted position is out of bounds")
                            tqdm.write(f"\tPredicted position: {xMostLikely + indexX}, {yMostLikely + indexY}, Predicted values: {predXY}")
                        pass
                    else:
                        Predictions[xMostLikely + indexX, yMostLikely + indexY] += circleWeights[xCircle+2,yCircle+2] #for some reason the reconstruction is transposed. Is adjusted later when plotted
        allKnownPositions += [(XCoordsLocal[i].item() + indexX, YCoordsLocal[i].item() + indexY) for i in range(len(XCoordsLocal))]
    
del zernikeValuesTotal

groundTruthCalculator(RelLabelCSV, groundTruth)


for cnt, posVal in tqdm(enumerate(Predictions.flatten()), desc = "Getting all known positions", total = Predictions.shape[0]*Predictions.shape[1]):
    x = int(np.round(cnt//Predictions.shape[1]))
    y = int(np.round(cnt%Predictions.shape[1]))
    
    xBefore = min(x,15)
    xAfter = min(Predictions.shape[0]-x,15)
    yBefore = min(y,15)
    yAfter = min(Predictions.shape[1]-y,15)
    
    posValScaler = (xBefore+xAfter)*(yBefore+yAfter)/30/30

    #TODO: remove
    #commented out so that all positions are considered
    # if posVal < 9*9*3*posValScaler+1: #9 times 9 predictions are looking at every pixel, factor 2 because of the circle
    #     continue
    if (x,y) in allKnownPositions:
        continue
    allKnownPositions.append((x,y))

#remove duplicates from allKnownPositions
allKnownPositions = list(set(allKnownPositions))
plotGTandPred(atomStruct, groundTruth, Predictions, start, end, allKnownPositions)
print(f"number of all known positions: {len(allKnownPositions)}")
print(f"number of all positions: {Predictions.shape[0]*Predictions.shape[1]}")
del groundTruth
del Predictions

allKnownPositionsInA = []
allKnownPositionsMeasurements = []
xRange = np.max(np.array(allKnownPositions)[:,0]) 
yRange = np.max(np.array(allKnownPositions)[:,1]) 
for pos in allKnownPositions:
    allKnownPositionsInA.append(((pos[0]-xRange/2)*0.2*1e-10,(pos[1]-yRange/2)*0.2*1e-10))
    allKnownPositionsMeasurements.append(measurementArray[pos[0],pos[1]])



del allKnownPositions

fullLengthY = measurementArray.shape[1]
fullLengthX = measurementArray.shape[0]
print("Saving the measurementArray and realPositions")
with open('measurementArray.npy', 'wb') as f:

    np.save(f, measurementArray)
    np.save(f, realPositions)
del measurementArray
del realPositions
del ZernikeObject
print("Finished saving the measurementArray and realPositions")



allKnownPositionsMeasurements = np.array(allKnownPositionsMeasurements) 
allKnownPositionsMeasurements /= np.sum(allKnownPositionsMeasurements)/len(allKnownPositionsMeasurements)
measurementArrayToFile = np.ravel(allKnownPositionsMeasurements)

#create ROP Files which take in the predicted positions
createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile,
               allKnownPositionsInA, diameterBFDNotScaled, fullLengthX = fullLengthX, 
               fullLengthY = fullLengthY, folder = "PredROP", grid = False)

del allKnownPositionsInA
del allKnownPositionsMeasurements
del measurementArrayToFile


with open('measurementArray.npy', 'rb') as f:

    measurementArray = np.load(f, allow_pickle=True)
    realPositions = np.load(f, allow_pickle=True)

measurementArray /= np.sum(measurementArray)/len(realPositions)
measurementArrayToFile = np.ravel(measurementArray)

#create ROP Files which takes in all positions but they are specified
createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile,
               realPositions, diameterBFDNotScaled, fullLengthX = fullLengthX,
               fullLengthY = fullLengthY, folder = "TotalPosROP", grid = False)
#create ROP Files which take in all positions but its using the grid Scan
createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile,
               realPositions, diameterBFDNotScaled, fullLengthX = fullLengthX,
               fullLengthY = fullLengthY, folder = "TotalGridROP", grid = True)