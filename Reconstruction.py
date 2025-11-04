import argparse
import glob
import os
import shutil
import struct
import cv2
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from FullPixelGridML.SimulateTilesOneFile import generateDiffractionArray, saveAllPosDifPatterns, generate_sparse_grid
import numpy as np
from Zernike.ScansToAtomsPosNN import TwoPartLightning, ThreePartLightningVIT





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

def writeParamsCNF(ScanX,ScanY, beamPositions, diameterBFD, conv_angle_in_mrad= 33, energy = 60e3, CBEDDim=50, folder = ".", grid = False, defocus = 0):
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

    #defocspread:   0
    replace_line_in_file(f'{folder}/Params.cnf', 5, f"E0: {energy:.0e}        #defocus in m")
    replace_line_in_file(f'{folder}/Params.cnf', 22, f"defocspread: {defocus}e-10        #Acceleration voltage")
    replace_line_in_file(f'{folder}/Params.cnf', 28, f"ObjAp:         {conv_angle}    #Aperture angle in Rad")
    replace_line_in_file(f'{folder}/Params.cnf', 31, f"ObjectDim:  {objectDim}  #Object dimension")
    replace_line_in_file(f'{folder}/Params.cnf', 32, f"ProbeDim:  {probeDim}  #Probe dimension")
    replace_line_in_file(f'{folder}/Params.cnf', 33, f"PixelSize: {realSpacePixelSize_in_A:.6}e-10 #Real space pixel size")
    replace_line_in_file(f'{folder}/Params.cnf', 37, f"CBEDDim:  {CBEDDim}  #CBED dimension")
    replace_line_in_file(f'{folder}/Params.cnf', 40, f"ptyMode: {int(not grid)}      #Input positions: 0 == grid scan; 1 == arbitrary (need to be specified below)")
    replace_line_in_file(f'{folder}/Params.cnf', 44, f"ScanX:   {ScanX}     #Number of steps in X-direction")
    replace_line_in_file(f'{folder}/Params.cnf', 45, f"ScanY:   {ScanY}     #Number of steps in Y-direction")
    replace_line_in_file(f'{folder}/Params.cnf', 46, f"batchSize: {min(ScanX,ScanY)}  #Number of CBEDs that are processed in parallel")



def createMeasurementArray(energy, conv_angle_in_mrad, structure, start, end, defocus, DIMTILES = 15, onlyPred = False, device = "gpu", noise : float = 0):
    nameStruct, gridSampling, atomStruct, measurement_thick , potential_thick = generateDiffractionArray(conv_angle= conv_angle_in_mrad, 
                                                                                                         energy=energy,structure=structure,
                                                                                                         pbar = True, start=start, end=end, 
                                                                                                         device=device, deviceAfter="cpu", defocus = defocus, noise = noise)
    # assert(measurement_thick.array.shape[-2] == measurement_thick.array.shape[-1])
    print(f"nameStruct: {nameStruct}")
    CBEDDim = min(measurement_thick.array.shape[-2:])
    print(f"CBEDDim: {CBEDDim}")
    measurementArray = np.zeros((measurement_thick.array.shape[0],measurement_thick.array.shape[1],CBEDDim,CBEDDim))
     
    realPositions = np.zeros((measurementArray.shape[0],measurementArray.shape[1]), dtype = object)   
    allCoords = np.copy(realPositions)

    for i in tqdm(range(measurementArray.shape[0]), desc="Creating measurement array"): 
        if not onlyPred: computetRow = measurement_thick.array[i,:,:,:].compute()
        for j in range(measurementArray.shape[1]): 
            realPositions[i,j] = ((j-measurementArray.shape[1]/2)*0.2*1e-10,(i-measurementArray.shape[0]/2)*0.2*1e-10)
            allCoords[i,j] = (i,j)  
            if not onlyPred: measurementArray[i,j,:,:] = cv2.resize(computetRow[j,:,:], dsize=(CBEDDim, CBEDDim), interpolation=cv2.INTER_LINEAR)      
    if not onlyPred: measurementArray[0,0,:,:] = cv2.resize(measurement_thick.array[0,0,:,:].compute(), dsize=(CBEDDim, CBEDDim), interpolation=cv2.INTER_LINEAR)
    # for BFD calculation

    assert(measurementArray.shape[0] > DIMTILES)
    assert(measurementArray.shape[1] > DIMTILES)
    return nameStruct,gridSampling,atomStruct,measurement_thick,potential_thick,CBEDDim,measurementArray, realPositions.flatten(), allCoords.flatten()



        #measurementArray[i,j,:,:] = measurementArray[i,j,:,:]/(np.sum(measurementArray[i,j,:,:])+1e-10) 



def CleanUpROP():
    #remove ROP clutter
    folders = [".","TotalPosROP","TotalGridROP"]
    folders += [f"PredROP{i}" for i in range(2,10)]
    folders += [f"PredROP{i}_rndm" for i in range(2,10)]
    for folder in folders:
        for filename in glob.glob(f"{folder}/Probe*.bin"):
            os.remove(f"{filename}")
        for filename in glob.glob(f"{folder}/Potential*.bin"):
            os.remove(f"{filename}")
        for filename in glob.glob(f"{folder}/Positions*.txt"):
            os.remove(f"{filename}")
        for filename in glob.glob(f"{folder}/Potential*.png"):
            os.remove(f"{filename}")
        if folder != ".": 
            for filename in glob.glob(f"{folder}/slurm*.out"):
                os.remove(f"{filename}")

def createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile, allKnownPositionsInA, diameterBFD, fullLengthX, fullLengthY,defocus, folder = ".", grid = False):
    writeParamsCNF(fullLengthX,fullLengthY, allKnownPositionsInA, diameterBFD, conv_angle_in_mrad = conv_angle_in_mrad, energy=energy, CBEDDim=CBEDDim, folder = folder, grid = grid, defocus=defocus)
    size = np.prod(measurementArrayToFile.shape)
    #Normalize data - required for ROP - now done before
    # measurementArrayToFile /= (np.sum(measurementArrayToFile)/(measurementArrayToFile.shape[0] * measurementArrayToFile.shape[1]))
    # measurementArrayToFile /= np.sum(measurementArrayToFile)/len(allKnownPositionsMeasurements)
    #Convert to binary format
    # for name in dir():
    #     try:
    #         if not name.startswith('_'):
    #             print(f"Name: {name}")
    #             print(f"\tsize: {asizeof.asizeof(eval(name))}")
    #     except:
    #         pass
    
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

def plotGTandPred(atomStruct, groundTruth, Predictions, start, end, makeNewFolder, predictions_scored_by_gaussian):
    """Create plots of the ground truth and the predictions. 
    They are transposed because the reconstruction is also transposed.
    """
    if not makeNewFolder:
        makeNewFolder = ""
    else:
        makeNewFolder+="/"

    extent=  (start[0],end[0],start[1],end[1])
    plt.imshow(Predictions.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig(makeNewFolder + "Predictions.png")
    plt.close()
    plt.imsave(makeNewFolder + "Predictions_No_Border.png", Predictions.T, origin="lower")
    plt.imshow(np.log(Predictions+1).T, origin = "lower", extent=extent)
    plt.savefig(makeNewFolder + "PredictionsLog.png")
    plt.close()

    
    plt.imshow(np.log(Predictions+1).T, origin = "lower", extent=extent)
    plt.savefig(makeNewFolder + "PredictionsLog.png")
    plt.close()
    import abtem
    abtem.show_atoms(atomStruct, legend = True)
    plt.savefig(makeNewFolder + "testStructureOriginal.png")
    plt.close()
    plt.imshow(groundTruth.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig(makeNewFolder + "groundTruth.png")


    # from matplotlib import cm 
    # cm1 = cm.get_cmap('RdBu_r')
    # cm1.set_under('white')
    # fig, ax = plt.subplots()
    # ax.imshow(groundTruth.T, origin = "lower", interpolation="none", extent=extent, cmap=cm1, vmin=12, alpha=0.5)
    # grid = np.zeros_like(groundTruth.T)
    # grid[::3, ::3] = 1
    # ax.imshow(grid, alpha=0.5)
    # ax.set_axis_off()
    # plt.savefig("groundTruth.png")

    plt.close()
    plt.imshow(np.log(groundTruth+1).T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig("groundTruthLog.png")
    plt.close()

    
    plt.imshow(predictions_scored_by_gaussian.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig(makeNewFolder + "predictions_scored_by_gaussian.png")
    plt.close()

    predictions_scored_by_gaussian_round_to_lower_1000 = np.floor(predictions_scored_by_gaussian/1000)*1000
    plt.imshow(predictions_scored_by_gaussian_round_to_lower_1000.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig(makeNewFolder + "predictions_scored_by_gaussian_rounded.png")
    plt.close()


# def createAndPlotReconstructedPotential(potential_thick, start, end, measurement_thick, conv_angle_in_mrad, energy):
#     multislice_reconstruction_ptycho_operator = RegularizedPtychographicOperator(
#     measurement_thick,
#     scan_step_sizes = 0.2,
#     semiangle_cutoff=conv_angle_in_mrad,
#     energy=energy,
#     #num_slices=10,
#     device="gpu",
#     #slice_thicknesses=1
# ).preprocess()
#     mspie_objects, mspie_probes, rpie_positions, mspie_sse = multislice_reconstruction_ptycho_operator.reconstruct(
#     max_iterations=20, return_iterations=True, random_seed=1, verbose=True)

#     mspie_objects[-1].angle().interpolate(potential_thick.sampling).show(extent = [start[0],end[0],start[1],end[1]])
#     plt.tight_layout()
#     plt.savefig("testStructureFullRec.png")

def groundTruthCalculator(LabelCSV, groundTruth, makeNewFolder):
    # create file for YOLO training
    if makeNewFolder:
        f =  open(makeNewFolder + "/" + "groundTruth.txt", 'w+') 
    #label, x (fraction of total image), y (fraction of total image), width (fraction of total image), height (fraction of total image)
    for i in range(len(LabelCSV)//3):
        gtDist = [LabelCSV[i*3], LabelCSV[1 + i*3]]
        xGT = int(np.around(float(gtDist[0])))
        yGT = int(np.around(float(gtDist[1])))
        element = int(float(LabelCSV[2+i*3]))
        # print(f"atoms no. {i}: gt element: {element}, Ground truth position: {xGT}, {yGT}, Ground truth values: {gtDist}")
        if xGT < 0 or xGT >= groundTruth.shape[0] or yGT < 0 or yGT >= groundTruth.shape[1]:
            # print("Ground truth out of bounds")
            pass
        else:
            groundTruth[int(xGT), int(yGT)] += 1#element
            if makeNewFolder:
                element_size_in_A = 1
                fractionXPosTotalImage = xGT / groundTruth.shape[0]
                fractionYPosTotalImage = yGT / groundTruth.shape[1] 
                fractionXPosTotalImage = np.around(fractionXPosTotalImage, decimals=5)
                fractionYPosTotalImage = np.around(fractionYPosTotalImage, decimals=5)
                fractionXWidthTotalImage = element_size_in_A * 5 / groundTruth.shape[0]
                fractionYHeightTotalImage = element_size_in_A * 5 / groundTruth.shape[1] 
                print(f"0 {fractionXPosTotalImage} {fractionYPosTotalImage} {fractionXWidthTotalImage} {fractionYHeightTotalImage}", file=f)





def createPredictionsWithFiles(indicesSortedByHighestPredicitionMinusInitial, energy, conv_angle_in_mrad, start, end, atomStruct, CBEDDim, measurementArray, allCoords, diameterBFDNotScaled,
                               groundTruth, chosenCoords2d, numberOfPosIndex, defocus, extent, rndm = False):
    folder = f"PredROP{numberOfPosIndex}_rndm" if rndm else f"PredROP{numberOfPosIndex}"
    print(f"\n\nCalculating PredROP{numberOfPosIndex}" + ("_rndm" if rndm else "") )
    allKnownPositions = []
    Position_Mask = np.zeros_like(groundTruth)
    for x, y in tqdm(allCoords, "Going through all known positions"):
        if (not rndm and ((x,y) in chosenCoords2d)) or (x,y) in indicesSortedByHighestPredicitionMinusInitial:
            allKnownPositions.append((x,y))
            Position_Mask[x,y] = 1
    plt.imshow(Position_Mask.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig(f"{folder}/PosMask.png")
    plt.close()
        
    GTinAllKnown = np.zeros_like(groundTruth)
    for pos in allKnownPositions:
        GTinAllKnown[pos] = groundTruth[pos]
    plt.imshow(GTinAllKnown.T, origin = "lower", interpolation="none", extent=extent)
    plt.colorbar()
    plt.savefig(f"{folder}/GTTimesPred.png")
    plt.close()
    print("Finished")
    #remove duplicates from allKnownPositions
    #allKnownPositions = list(set(allKnownPositions))
    
    print(f"number of all known positions: {len(allKnownPositions)}")

    allKnownPositionsInA = []
    allKnownPositionsMeasurements = []
    xRange = np.max(np.array(allKnownPositions)[:,0]) 
    yRange = np.max(np.array(allKnownPositions)[:,1]) 
    for pos in allKnownPositions:
        allKnownPositionsInA.append(((pos[1]-yRange/2)*0.2*1e-10, (pos[0]-xRange/2)*0.2*1e-10))
        allKnownPositionsMeasurements.append(measurementArray[pos[0],pos[1],:,:])



    del allKnownPositions

    fullLengthY = measurementArray.shape[1]
    fullLengthX = measurementArray.shape[0]
    # print("Saving the measurementArray and realPositions")
    # with open('measurementArray.npy', 'wb') as f:

    #     np.save(f, measurementArray)
    #     np.save(f, realPositions)
    # del measurementArray
    # del realPositions

    # print("Finished saving the measurementArray and realPositions")



    allKnownPositionsMeasurements = np.array(allKnownPositionsMeasurements) 
    allKnownPositionsMeasurements /= np.sum(allKnownPositionsMeasurements)/len(allKnownPositionsMeasurements)
    measurementArrayToFile = np.ravel(allKnownPositionsMeasurements)
    
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass


    #create ROP Files which take in the predicted positions
    createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile,
                allKnownPositionsInA, diameterBFDNotScaled, fullLengthX = fullLengthX, 
                fullLengthY = fullLengthY ,defocus=defocus
                , folder = folder, grid = False)

    del allKnownPositionsInA
    del allKnownPositionsMeasurements
    del measurementArrayToFile
    return fullLengthY,fullLengthX


def CreatePredictions(resultVectorLength, model, PredShape, zernikeValuesUnsparsed, chosenCoords, scalerEnabled, windowSizeInCoords, nonPredictedBorderinCoordinates, zernike = True):
    #load scaling factors

    if scalerEnabled:


        with open('Zernike/stdValues.csv', 'r') as file:
            line = file.readline()
            stdValues = [float(x.strip()) for x in line.split(',') if x.strip()] #TODO es werden hier auch die Koordinaten mitgeladen, raus wenn im dataset raus
            stdValuesArray = torch.tensor(stdValues[:resultVectorLength] + stdValues[-2:]) 
            stdValuesArray = torch.clamp(stdValuesArray, min=1e-5, max=None) #avoid division by zero #TODO hier wird jetzt geclipt, im dataset neuerdings auch
        with open('Zernike/meanValues.csv', 'r') as file:
            line = file.readline()
            meanValues = [float(x.strip()) for x in line.split(',') if x.strip()]
            meanValuesArray : torch.Tensor = torch.tensor(meanValues[:resultVectorLength]  + meanValues[-2:])  #TODO es werden hier auch die Koordinaten mitgeladen, raus wenn im dataset raus

    # loop over the all positions and apply the model to the data
    Predictions = np.zeros(PredShape)
    PositionsToScanX = chosenCoords[:,0]
    PositionsToScanY = chosenCoords[:,1]
    

    

    imageOrZernikeMoments = None
    for indexX in tqdm(range(-windowSizeInCoords ,Predictions.shape[0] + 1), desc  = "Going through all positions and predicting"):
        for indexY in range(-windowSizeInCoords ,Predictions.shape[1] + 1):
            PositionsToScanXLocal = PositionsToScanX - indexX
            PositionsToScanYLocal = PositionsToScanY - indexY
            mask = (PositionsToScanXLocal >= -nonPredictedBorderinCoordinates) * (PositionsToScanYLocal >= - nonPredictedBorderinCoordinates) * (PositionsToScanXLocal <= (windowSizeInCoords + nonPredictedBorderinCoordinates)) * (PositionsToScanYLocal <= (windowSizeInCoords + nonPredictedBorderinCoordinates))
            numberOfAtomsInWindow = mask.sum()
            if numberOfAtomsInWindow <= 1:
                continue
            PositionsToScanXLocal = PositionsToScanXLocal[mask]
            PositionsToScanYLocal = PositionsToScanYLocal[mask]
            YCoordsLocal = torch.tensor(PositionsToScanYLocal)
            XCoordsLocal = torch.tensor(PositionsToScanXLocal)
            # print(f"XCoordsLocal: {XCoordsLocal}")
            # print(f"YCoordsLocal: {YCoordsLocal}")
        
            # imageOrZernikeMoments = zernikeValuesWindow.reshape((zernikeValuesWindowSizeX,zernikeValuesWindowSizeY,-1))[XCoordsLocal + min(indexX,0),YCoordsLocal + min(indexY,0)].reshape((XCoordsLocal.shape[0],-1))
            try: 
                imageOrZernikeMoments = zernikeValuesUnsparsed[(indexX + XCoordsLocal), (indexY + YCoordsLocal), :].reshape(-1, resultVectorLength)
            except IndexError as E:
                print("Error: Index out of bounds")
                print(f"XCoordsLocal: {XCoordsLocal}")
                print(f"YCoordsLocal: {YCoordsLocal}")
                print(f"indexX: {indexX}")
                print(f"indexY: {indexY}")
                raise Exception(E)
            imageOrZernikeMoments = torch.cat((imageOrZernikeMoments, torch.stack([XCoordsLocal, YCoordsLocal]).T), dim = 1)
            if scalerEnabled: imageOrZernikeMoments = (imageOrZernikeMoments - meanValuesArray[np.newaxis,:]) / stdValuesArray[np.newaxis,:] 
            #y and x are now correct
            imageOrZernikeMomentsCuda = imageOrZernikeMoments.to(torch.device("cuda"))

            with torch.inference_mode():
                pred = model(imageOrZernikeMomentsCuda.unsqueeze(0))
                pred = pred.cpu().detach().numpy().reshape((1,-1,2))[0]
            #TODO scaler wieder einfügen? Macht Randpositionen vielleicht besser
            # predRealXCoord  = np.round(pred[:,0]).astype(int) + indexX
            # predRealYCoord  = np.round(pred[:,1]).astype(int) + indexY
            # predRealCoordInside = (predRealXCoord >= 0) * (predRealYCoord >= 0) * (predRealXCoord < Predictions.shape[0]) * (predRealYCoord < Predictions.shape[1])
            # #(zernikeValuesWindowSizeX/15 * zernikeValuesWindowSizeY/15 / (predRealCoordInside.sum()/10)) #10 is the number of predictions, 15 is the window size, only make as many predictions as there are 
            scaler = 1                                                                                          #positions inside

            for predXY in pred:
                for xCircle in range(-1,2):
                    for yCircle in range(-1,2):                  
                        currentX = np.round(predXY[0]).astype(int)+xCircle
                        currentY = np.round(predXY[1]).astype(int)+yCircle
                        weight = 1 - np.min((np.square(predXY[0] - currentX) + np.square(predXY[1] - currentY), 1))
                        currentNonRelX = indexX + currentX
                        currentNonRelY = indexY + currentY
                        if currentNonRelX < 0 or currentNonRelX >= Predictions.shape[0] or currentNonRelY < 0 or currentNonRelY >= Predictions.shape[1]:
                        #skip if the predicted position is outside the sample
                            pass
                        else:
                            Predictions[currentNonRelX, currentNonRelY] += weight
    return Predictions


def rescalePredicitions(Predictions, windowSizeInCoords = 15):
    def rescaleToInner(Image,minInner, maxInner):
        Image = Image - np.min(Image)
        scaler = np.max(Image) - np.min(Image)
        scaler = scaler or 1
        Image = Image / scaler * (maxInner - minInner) + minInner
        return Image


    noiseLevelInner = np.median(Predictions[windowSizeInCoords:-windowSizeInCoords,windowSizeInCoords:-windowSizeInCoords])
    maxInner = np.max(Predictions[windowSizeInCoords:-windowSizeInCoords,windowSizeInCoords:-windowSizeInCoords])
    minInner = np.min(Predictions[windowSizeInCoords:-windowSizeInCoords,windowSizeInCoords:-windowSizeInCoords])
    for outerEdge in range(1,windowSizeInCoords):
        noiseLevelUp = np.median(Predictions.T[outerEdge,outerEdge:-outerEdge])
        noiseLevelDown = np.median(Predictions.T[-outerEdge,outerEdge:-outerEdge])
        noiseLevelLeft = np.median(Predictions[outerEdge,outerEdge:-outerEdge])
        noiseLevelRight = np.median(Predictions[-outerEdge,outerEdge:-outerEdge])
    
        Predictions.T[outerEdge,outerEdge:-outerEdge] -= noiseLevelUp - noiseLevelInner
        Predictions.T[-outerEdge,outerEdge:-outerEdge] -= noiseLevelUp - noiseLevelInner
        Predictions[outerEdge,outerEdge:-outerEdge] -= noiseLevelLeft - noiseLevelInner
        Predictions[-outerEdge,outerEdge:-outerEdge] -= noiseLevelRight - noiseLevelInner

    # Predictions.T[outerEdge,outerEdge:-outerEdge] = rescaleToInner(Predictions.T[outerEdge,outerEdge:-outerEdge],minInner, maxInner)
    # Predictions.T[-outerEdge,outerEdge:-outerEdge] = rescaleToInner(Predictions.T[-outerEdge,outerEdge:-outerEdge],minInner, maxInner)
    # Predictions[outerEdge,outerEdge:-outerEdge] = rescaleToInner(Predictions[outerEdge,outerEdge:-outerEdge],minInner, maxInner)
    # Predictions[-outerEdge,outerEdge:-outerEdge] = rescaleToInner(Predictions[-outerEdge,outerEdge:-outerEdge],minInner, maxInner)

    noiseLevelUp = np.median(Predictions.T[0,:])
    noiseLevelDown = np.median(Predictions.T[0,:])
    noiseLevelLeft = np.median(Predictions[0,:])
    noiseLevelRight = np.median(Predictions[0,:])

    Predictions.T[0,:] -= noiseLevelUp - noiseLevelInner
    Predictions[0,:] -= noiseLevelLeft - noiseLevelInner
    return Predictions

from scipy.ndimage import gaussian_filter

def greedy_gaussian_suppression(pred_map, chosenCoords2d, num_points=20, suppression_sigma=3.0, suppression_strength=1.0):
    """
    Selects top-scoring pixels with spatial diversity using greedy selection with Gaussian suppression.

    Parameters:
        pred_map (np.ndarray): 2D prediction map.
        chosenCoords2d (np.ndarray): List of (y, x) coordinates already visited (will be masked).
        num_points (int): Number of top points to select.
        suppression_sigma (float): Sigma for Gaussian suppression kernel.
        suppression_strength (float): Multiplier for how much to suppress (1.0 = full subtraction).

    Returns:
        selected_indices (list of tuples): Selected (y, x) indices.
    """
    pred = pred_map.copy()
    
    # Mask already scanned positions
    for x, y in chosenCoords2d:
        pred, exit = gaussian_suppressor(np.s_[x,y], suppression_sigma, 1, pred, radius=3)

    selected_indices = []
    for _ in tqdm(range(num_points), desc="Greedy selection with Gaussian suppression on the prediction"):
        max_idx = np.unravel_index(np.argmax(pred), pred.shape)
        pred, exit = gaussian_suppressor(max_idx, suppression_sigma, suppression_strength, pred, radius = 3)
        if exit is True:
            break
        selected_indices.append(max_idx)
    return selected_indices

def gaussian_suppressor(idx, suppression_sigma, suppression_strength, pred, radius):
    val_at_idx = pred[idx]
    
    if val_at_idx == -np.inf:
        return pred, True #no more valid points

    # Create Gaussian mask centered at max_idx
    mask = np.zeros_like(pred)
    mask[idx] = suppression_strength * val_at_idx
    gaussian_mask = gaussian_filter(mask, sigma=suppression_sigma, radius = radius)

    # Subtract the Gaussian mask to suppress surrounding area
    pred = pred - gaussian_mask
    pred[idx] = -np.inf
    return pred, False

from math import ceil, sqrt

def generate_even_grid(shape, num_points):
    """
    Generate approximately evenly spaced (x, y) grid points over a 2D array.
    
    Parameters:
        shape (tuple): (height, width) of the 2D array.
        num_points (int): Number of points to generate.
    
    Returns:
        List of (x, y) indices (rounded to nearest integers).
    """
    h, w = shape

    # Estimate number of rows and cols to get close to num_points
    aspect_ratio = w / h
    n_rows = int(round(sqrt(num_points / aspect_ratio)))
    n_cols = int(ceil(num_points / n_rows))

    # Ensure we stay within bounds
    ys = np.linspace(0, h - 1, n_rows)
    xs = np.linspace(0, w - 1, n_cols)

    # Cartesian product
    grid = [(int(round(x)), int(round(y))) for x in xs for y in ys]

    # Return exactly num_points (may slice off extras)
    return grid[:num_points]

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruction script for ptychographic data.")
    parser.add_argument('--structure', type=str, default="/data/scratch/haertelk/Masterarbeit/FullPixelGridML/structures/used/MoS2_hexagonal.cif",
                        help='Path to the structure .cif file')
    parser.add_argument('--model', type=str, default="/data/scratch/haertelk/Masterarbeit/checkpoints/DQN_0503_1255_Z_TransfEnc_9Pos_5hidlay_9000E/epoch=1169-step=49140.ckpt",
                        help='Path to the model checkpoint')
    parser.add_argument('--sparseGridFactor', type=int, default=4,
                        help='Sparse grid factor for Zernike moment calculation')
    parser.add_argument('--makeNewFolder', type=str, default="",help ="If enabled, uses a new folder for the results in ROP Results.")
    parser.add_argument('--defocus', type=int, default=0,
                        help='Defocus value for the reconstruction, default is 0.')
    parser.add_argument("--noScaler", action="store_true", help="If enabled, the zernike moments are NOT scaled before usage. Only for old model")
    parser.add_argument('--onlyPred', action="store_true", help='If enabled, only the predictions are calculated without the full reconstruction.')
    parser.add_argument("--nonPredictedBorderinCoordinates", "-b",type=int, default=15, help="The border in coordinates that is not predicted. Default is 15.")
    parser.add_argument('--pixel', action="store_true", help='If enabled, Zernike moments are not used. Instead the pixel representation is used.')
    
    return parser.parse_args()

def BFD_calculation(onlyPred, diameterBFD50Pixels, measurement_thick, measurementArray):
    diameterBFDNotScaled = diameterBFD50Pixels / 50 * min((measurementArray.shape[-1], measurementArray.shape[-2])) #diameterBFD50Pixels is the diameter in pixels, not in nm
    if not onlyPred:
        from Zernike.ZernikeTransformer import calc_diameter_bfd

        diameterBFDNotScaled = calc_diameter_bfd(measurement_thick.array[0,0,:,:].compute())
        print(f"Calculated BFD not scaled to 50 pixels: {diameterBFDNotScaled}")
        print(f"Calculated BFD scaled to 50 pixels: {diameterBFDNotScaled/min((measurementArray[0,0,:,:].shape[-1], measurementArray[0,0,:,:].shape[-2])) * 50}")
    plt.imsave("detectorImage.png",measurement_thick.array[0,0,:,:].compute())


    print(f"imageDim Not Scaled to 50 Pixels: {min((measurementArray[0,0,:,:].shape[-1], measurementArray[0,0,:,:].shape[-2]))}")

    #print(f"Actually used BFD with margin: {diameterBFD50Pixels}")
    return diameterBFDNotScaled

    # The result vector length is the number of OSA-ANSI moments times the number of atoms

def createEmptyBackground(diameterBFD50Pixels, zernike, resultVectorLength, chosenCoords2dArray,
                          model, scalerEnabled, windowSizeInCoords, xMaxCNT, yMaxCNT,
                          nonPredictedBorderinCoordinates, emptyInput = False):
    

    
    if zernike: 
        from Zernike.ZernikePolynomials import Zernike
        ZernikeObject = Zernike(resultVectorLength)
        #raise Exception("New empty zernike pattern with OSA have to be calculated")
        #empty zernike pattern.npy is old
        empty_background_image = np.load("diffraction_pattern_full_size_empty_space.npy", allow_pickle=True)[None,:,:]
        empty_background_image = ZernikeObject.zernikeTransform(dataSetName = None, groupOfPatterns = empty_background_image, hdf5File = None)[0]
        
        if emptyInput: empty_background_image = np.zeros_like(empty_background_image)
    else: 
        empty_background_image = np.load("diffraction_pattern_full_size_empty_space.npy", allow_pickle=True)
        dimNew = 100
        empty_background_image = cv2.resize(empty_background_image, dsize=(dimNew, dimNew), interpolation=cv2.INTER_LINEAR)
        BFDdiameterScaled = int(diameterBFD50Pixels * dimNew / 50)
        indicesInBFD = slice(max((dimNew - BFDdiameterScaled)//2-1,0),min((dimNew + BFDdiameterScaled)//2+1, dimNew ))
        empty_background_image = empty_background_image[indicesInBFD, indicesInBFD] 

    zernikeValuesUnsparsed = torch.zeros((xMaxCNT, yMaxCNT, resultVectorLength), dtype=torch.float32)
    for cnt, (x,y) in enumerate(chosenCoords2dArray):
        zernikeValuesUnsparsed[x, y, :] = torch.tensor(empty_background_image).float().reshape(resultVectorLength)           
    Predictions = CreatePredictions(resultVectorLength, model, (xMaxCNT, yMaxCNT), zernikeValuesUnsparsed, chosenCoords=chosenCoords2dArray, 
                                    scalerEnabled = scalerEnabled, windowSizeInCoords=windowSizeInCoords, 
                                    nonPredictedBorderinCoordinates = nonPredictedBorderinCoordinates, zernike = zernike)
    return Predictions
    




def parse_args_full():
    args = parse_args()
    structure = args.structure
    model_checkpoint = args.model
    print(f"Model: {model_checkpoint}")
    sparseGridFactor = args.sparseGridFactor
    print(f"Sparse grid factor: {sparseGridFactor}")
    defocus = args.defocus
    print(f"defocus: {defocus}")
    if defocus != 0 and model_checkpoint == "/data/scratch/haertelk/Masterarbeit/checkpoints/DQN_0503_1255_Z_TransfEnc_9Pos_5hidlay_9000E/epoch=1169-step=49140.ckpt":
        raise Exception("Using the old model, which does not use defocus.")
    makeNewFolder = args.makeNewFolder
    print(  f"Make new folder: {makeNewFolder}")
    onlyPred = args.onlyPred
    print(f"Only prediction: {onlyPred}")
    nonPredictedBorderinCoordinates = args.nonPredictedBorderinCoordinates
    print(f"Non predicted border in coordinates: {nonPredictedBorderinCoordinates}")
    if nonPredictedBorderinCoordinates > 0 and model_checkpoint == "/data/scratch/haertelk/Masterarbeit/checkpoints/DQN_0503_1255_Z_TransfEnc_9Pos_5hidlay_9000E/epoch=1169-step=49140.ckpt":
        raise Exception("Using the old model, which does not use non predicted border in coordinates.")
    zernike = not args.pixel
    print(f"Zernike moments enabled: {zernike}")
    scalerEnabled = not args.noScaler
    if zernike and scalerEnabled:
        print(f"Scaler enabled: {scalerEnabled}")
    elif not zernike and scalerEnabled:
        print("Scaler disabled because Zernike moments are not used.")
        scalerEnabled =False

    if scalerEnabled and model_checkpoint == "/data/scratch/haertelk/Masterarbeit/checkpoints/DQN_0503_1255_Z_TransfEnc_9Pos_5hidlay_9000E/epoch=1169-step=49140.ckpt":
        raise Exception("Using the old model, which does not use scaling for the Zernike moments.")
    elif not scalerEnabled and model_checkpoint != "/data/scratch/haertelk/Masterarbeit/checkpoints/DQN_0503_1255_Z_TransfEnc_9Pos_5hidlay_9000E/epoch=1169-step=49140.ckpt" and zernike:
        raise Exception("Using the new model, which uses scaling for the Zernike moments.")
    return structure,model_checkpoint,sparseGridFactor,defocus,makeNewFolder,scalerEnabled,onlyPred,nonPredictedBorderinCoordinates, zernike

def full_routine(structure, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates, energy= 60e3,
                 conv_angle_in_mrad = 33, diameterBFD50Pixels = 18, start = (5,5), end = (25,25), windowSizeInCoords = 15, numberOfAtoms = 9, numberOfOSAANSIMoments = 860, hidden_size_encoder = 1024,
                 zernike = True
                 ):
    print(f"Structure: {structure}")
    print(f"Number of atoms: {numberOfAtoms}")
    print(f"Window size in coordinates: {windowSizeInCoords}")
    if zernike:
        resultVectorLength = numberOfOSAANSIMoments
    else: 
        resultVectorLength = 38*38

    if not onlyPred: CleanUpROP()
    nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick, CBEDDim, measurementArray, realPositions, allCoords = createMeasurementArray(energy, conv_angle_in_mrad, structure, start, end,
                                                                                                                                                            defocus, onlyPred=onlyPred, device = "gpu") 


    diameterBFDNotScaled = BFD_calculation(onlyPred, diameterBFD50Pixels, measurement_thick, measurementArray)

    xMaxCNT, yMaxCNT = np.shape(measurement_thick.array)[:2] # type: ignore
    chosenCoords2d = generate_sparse_grid(xMaxCNT, yMaxCNT, sparseGridFactor, twoD=True)
    chosenCoords2dArray = np.array(chosenCoords2d)
    chosenCoords = np.array(generate_sparse_grid(xMaxCNT, yMaxCNT, sparseGridFactor, twoD=False))


    difArrays = [[nameStruct, gridSampling, atomStruct, measurement_thick, potential_thick]]
    del measurement_thick, potential_thick
    RelLabelCSV, zernikeValuesSparse = saveAllPosDifPatterns(None, -1, None, diameterBFD50Pixels,
                                                            processID = 99999, silence = False,
                                                            maxPooling = 1, structure = "None",
                                                            fileWrite = False, difArrays = difArrays, nonPredictedBorderInA=0,
                                                            start = start, end = end, zernike=zernike, initialCoords=chosenCoords,
                                                            defocus = defocus, numberOfOSAANSIMoments=numberOfOSAANSIMoments, step_size=100) # type: ignore
    RelLabelCSV, zernikeValuesSparse = RelLabelCSV[0][1:], zernikeValuesSparse[0]
    print(f"{zernikeValuesSparse.shape=}")

    zernikeValuesUnsparsed = torch.zeros((measurementArray.shape[0], measurementArray.shape[1], resultVectorLength), dtype=torch.float32)

    for cnt, (x,y) in enumerate(tqdm(chosenCoords2dArray, desc="Filling zernike values unsparsed")):
        zernikeValuesUnsparsed[x, y, :] = torch.tensor(zernikeValuesSparse[cnt, :]).float().reshape(resultVectorLength)

    if zernike:
        model = TwoPartLightning.load_from_checkpoint(checkpoint_path = model_checkpoint,
                                                    numberOfZernikeMoments = numberOfOSAANSIMoments,
                                                        numberOfAtoms = numberOfAtoms , hidden_size= hidden_size_encoder)
    else:
        model = ThreePartLightningVIT.load_from_checkpoint(checkpoint_path = model_checkpoint,
                                                        numberOfAtoms = numberOfAtoms, hidden_size= hidden_size_encoder)
    model.eval().to(torch.device("cuda"))

    Predictions = CreatePredictions(resultVectorLength, model, (xMaxCNT, yMaxCNT), zernikeValuesUnsparsed, chosenCoords=chosenCoords2dArray, scalerEnabled = scalerEnabled, windowSizeInCoords=windowSizeInCoords, nonPredictedBorderinCoordinates = nonPredictedBorderinCoordinates, zernike = zernike)
    emptyBackgroundPredictions = createEmptyBackground(diameterBFD50Pixels, zernike, resultVectorLength, chosenCoords2dArray,
                                                    model, scalerEnabled, windowSizeInCoords, xMaxCNT, yMaxCNT,
                                                    nonPredictedBorderinCoordinates)
    Predictions -= emptyBackgroundPredictions
    # Predictions = rescalePredicitions(Predictions, windowSizeInCoords = windowSizeInCoords)

    #create variable with the year, month, day, hour, minute and second of the current time
    from datetime import datetime
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


    if makeNewFolder:    
        makeNewFolder = f"ROPResults/{makeNewFolder}/{now}_{structure.split('/')[-1].split('.')[0]}_s{sparseGridFactor}_{model_checkpoint.split('/')[-2].split('.')[0]}"
        print("Creating new folder", makeNewFolder)
        os.makedirs(makeNewFolder, exist_ok=True)


    del zernikeValuesUnsparsed
    groundTruth = np.zeros((xMaxCNT, yMaxCNT))
    
    groundTruthCalculator(RelLabelCSV, groundTruth, makeNewFolder)
    indicesSortedByHighestPredicitionMinusInitial = greedy_gaussian_suppression(Predictions, chosenCoords2dArray, num_points=Predictions.shape[0]*Predictions.shape[1]- len(chosenCoords2dArray))
    predictions_scored_by_gaussian = np.zeros_like(Predictions)
    for cnt, (x,y) in enumerate(indicesSortedByHighestPredicitionMinusInitial):
        predictions_scored_by_gaussian[x,y] = len(indicesSortedByHighestPredicitionMinusInitial) - cnt

    
    plotGTandPred(atomStruct, groundTruth, Predictions, start, end, makeNewFolder, predictions_scored_by_gaussian)
    if makeNewFolder:
        return ""
    extent = (start[0],end[0],start[1],end[1])
    numberOfPositions = [0.5,0.4,0.3,0.2,0.1,0.08,0.07,0.063]
    for numberOfPosIndex, ratioPos in zip(range(2,10), numberOfPositions):
        numberOfAllPos = int(np.around(ratioPos*Predictions.shape[0]*Predictions.shape[1]))
        maxNumberOfPredicitions = max(numberOfAllPos - len(chosenCoords2dArray),0)
        numberOfPreditions = min((maxNumberOfPredicitions, len(indicesSortedByHighestPredicitionMinusInitial)))
        print(f"numberOfPreditions: {numberOfPreditions} in folder {numberOfPosIndex} with ratioPos {ratioPos}")
        fullLengthY, fullLengthX = createPredictionsWithFiles(indicesSortedByHighestPredicitionMinusInitial[:numberOfPreditions], energy, conv_angle_in_mrad, start, end, atomStruct, CBEDDim,
                                                              measurementArray, allCoords, diameterBFDNotScaled, groundTruth, chosenCoords2d, numberOfPosIndex, defocus, extent, rndm = False)

    #now create the ROP files but with just evenly spaced points instead of the predictions
    for numberOfPosIndex, ratioPos in zip(range(2,10), numberOfPositions):
        maxNumberOfPredicitions = int(np.around(ratioPos*Predictions.shape[0]*Predictions.shape[1]))
        evenlySpacedIndices = generate_even_grid(Predictions.shape, maxNumberOfPredicitions)
        fullLengthY, fullLengthX = createPredictionsWithFiles(evenlySpacedIndices, energy, conv_angle_in_mrad, start, end, atomStruct, CBEDDim, measurementArray, allCoords, diameterBFDNotScaled,
                                                              groundTruth, [], numberOfPosIndex, defocus, extent, rndm = True)

    if onlyPred:
        print("Only predictions were made, not the full reconstruction.")
        exit()

    del Predictions

    # with open('measurementArray.npy', 'rb') as f:

    #     measurementArray = np.load(f, allow_pickle=True)
    #     realPositions = np.load(f, allow_pickle=True)

    measurementArray /= np.sum(measurementArray)/len(realPositions)
    measurementArrayToFile = np.ravel(measurementArray)

    #create ROP Files which takes in all positions but they are specified
    createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile,
                realPositions, diameterBFDNotScaled, fullLengthX = fullLengthX,
                fullLengthY = fullLengthY, defocus=defocus, folder = "TotalPosROP", grid = False)
    #create ROP Files which take in all positions but its using the grid Scan
    createROPFiles(energy, conv_angle_in_mrad, CBEDDim, measurementArrayToFile,
                realPositions, diameterBFDNotScaled, fullLengthX = fullLengthX,
                fullLengthY = fullLengthY, defocus=defocus, folder = "TotalGridROP", grid = True)
    del measurementArrayToFile, realPositions
    del measurementArray


def main():
    structure, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates, zernike = parse_args_full()
    full_routine(structure, model_checkpoint, sparseGridFactor, defocus, makeNewFolder, scalerEnabled, onlyPred, nonPredictedBorderinCoordinates, zernike = zernike)

    print("Fertig")


if __name__ == "__main__":
    main()
