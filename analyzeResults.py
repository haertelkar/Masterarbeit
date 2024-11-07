import matplotlib.pyplot as plt
import csv
import numpy as np
import os

from tqdm import tqdm
from rowsIndexToHeader import headerToRowsIndexSingle as headerToRowsIndex
from datasets import ptychographicData
import pandas
import sys

def raiseExcep(header, index, file):
    if index >= 0:
        return index
    else:
        raise Exception(f"header '{header}' unknown in '{file}'. Skipped.")

    
def returnFullRowContent(fullRowIndices, fullRow, startInd, endInd):
    returnlist = []
    for i in range(startInd,endInd):
        if i in fullRowIndices:
            returnlist.append(fullRow[i])
    return returnlist

# def grabFileNames():
#     #only using it to grab file names, doesn't matter if it is Zernike or not
#     test_data = ptychographicData(
#                     os.path.abspath(os.path.join("FullPixelGridML", "measurements_test","labels.csv")), os.path.abspath(os.path.join("FullPixelGridML", "measurements_test"))
#                 ) 
#     fileNames = test_data.image_names
#     return fileNames

# fileNames = grabFileNames()
if len(sys.argv) > 1:
    onlyThisFile = sys.argv[1]
else:
    onlyThisFile = ""

for file in os.listdir(os.path.join(os.getcwd(), "testDataEval")):
    try:
        if "results" not in file or ".csv" != file[-4:] or not onlyThisFile in file:
            continue
        print(f"\nRun {file}")
        xGTs = {}
        xPreds = {}
        yGTs = {}
        yPreds = {}
        elementsGT = {}
        elementsPred = {}


        modelName = "_".join(file.split("_")[1:])
        modelName = ".".join(modelName.split(".")[:-1])

        csvFile = pandas.read_csv(os.path.join("testDataEval", file))
        for column in tqdm(csvFile.columns, desc = f"Going trough colums in file {file}", leave = False):
            if "_pred" in column or "pixel" in column:
                continue
            elif "xAtomRel" in column:
                atomNo = int(column.replace("xAtomRel",""))
                xGTs[atomNo] = np.array(csvFile[column]).astype(float)
                xPreds[atomNo] = np.array(csvFile[column + "_pred"]).astype(float)
            elif "Xcoord" in column:
                atomNo = int(column.replace("Xcoord",""))
                xGTs[atomNo] = np.array(csvFile[column]).astype(float)
                xPreds[atomNo] = np.array(csvFile[column + "_pred"]).astype(float)
            elif "yAtomRel" in column:
                atomNo = int(column.replace("yAtomRel",""))
                yGTs[atomNo] = np.array(csvFile[column]).astype(float)
                yPreds[atomNo] = np.array(csvFile[column + "_pred"]).astype(float)
            elif "Ycoord" in column:
                atomNo = int(column.replace("Ycoord",""))
                yGTs[atomNo] = np.array(csvFile[column]).astype(float)
                yPreds[atomNo] = np.array(csvFile[column + "_pred"]).astype(float)
            elif "element" in column:
                atomNo = int(column.replace("element",""))
                elementsGT[atomNo] = np.array(csvFile[column]).astype(float)
                elementsPred[atomNo] = np.array(csvFile[column + "_pred"]).astype(float)
            elif "elem" in column:
                atomNo = int(column.replace("elem",""))
                elementsGT[atomNo] = np.array(csvFile[column]).astype(float)
                elementsPred[atomNo] = np.array(csvFile[column + "_pred"]).astype(float)
        
        #get the distance between the atoms and the scan position
        for atomNo in xGTs.keys():
            xGT = xGTs[atomNo]
            xPred : np.ndarray = xPreds[atomNo]
            yGT = yGTs[atomNo]
            yPred = yPreds[atomNo]

            distancePredictionDelta = np.sqrt((xGT - xPred)**2 + (yGT - yPred)**2)
            distance = np.sqrt(xGT**2 + yGT**2)
            distanceToBeAccurate = 1

            plt.scatter(distance, distancePredictionDelta)
            plt.xlabel(f"Distance between atom nr.{atomNo} and scan position")
            plt.ylabel("Delta between distance prediction and actual distance")
            # plt.legend()
            plt.savefig(os.path.join("DeltaToDistance", file + f"distanceToAccuracy_atom{atomNo}_labeled.png"))
            plt.close()
            binDistance = int((max(distance) - min(distance)) // distanceToBeAccurate)
            binDistanceDelta = int((max(distancePredictionDelta) - min(distancePredictionDelta)) // distanceToBeAccurate)
            if binDistance == 0 or binDistanceDelta == 0:
                print(f"All predicted distances are the same for atom nr.{atomNo}. Skipped.")
                continue
            heatmap, xedges, yedges = np.histogram2d(distance, distancePredictionDelta, bins=(binDistance, binDistanceDelta))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            logHeatmap = np.log(heatmap.T, where=heatmap.T != 0)
            logHeatmap[heatmap.T == 0] = np.log(1) - 1 #to make zero the lowest value
            plt.clf()
            plt.imshow(logHeatmap, extent=extent, origin='lower')
            plt.savefig(os.path.join("DeltaToDistance", file + f"dTdeltaHeatmap_atom{atomNo}_labeled.png"))
            plt.close()
            plt.hist(distancePredictionDelta, bins = 100)
            plt.savefig(os.path.join("DeltaToDistance", file + f"hist_atom{atomNo}_labeled.png"))
            plt.close()
            print(f"Distance between atom nr.{atomNo} and scan position less than {distanceToBeAccurate} in {100*len(distance[np.array(distancePredictionDelta) < distanceToBeAccurate])/len(distance):.2f}% of cases over all structure")
            print(f"Distance between atom nr.{atomNo} and scan position less than {distanceToBeAccurate}*2 in {100*len(distance[np.array(distancePredictionDelta) < 2*distanceToBeAccurate])/len(distance):.2f}% of cases over all structure")
            print(f"Distance between atom nr.{atomNo} and scan position less than {distanceToBeAccurate}*3 in {100*len(distance[np.array(distancePredictionDelta) < 3*distanceToBeAccurate])/len(distance):.2f}% of cases over all structure")
        if len(xGTs) == 0:
            print(f"No atoms found in {file}. Skipped.")
            continue
        for cnt in range(len(xGTs[0])//10):
            if cnt % 150 != 0:
                continue
            
            predGrid = np.zeros((15,15))
            GTGrid = np.zeros((15,15))
            for atomNo in xGTs.keys():
                xGT = xGTs[atomNo]
                xPred = xPreds[atomNo]
                yGT = yGTs[atomNo]
                yPred = yPreds[atomNo]
                predGrid[np.clip(np.round(xPred[cnt]),0,14).astype(int), np.clip(np.round(yPred[cnt]),0,14).astype(int)] = 1
                GTGrid[np.clip(np.round(xGT[cnt]),0,14).astype(int), np.clip(np.round(yGT[cnt]),0,14).astype(int)] = 1
            
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(predGrid)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(GTGrid)
            plt.colorbar()
            plt.suptitle(f"Model: {modelName}")
            plt.savefig(os.path.join("testDataEval", file + f"row{cnt}_comparison.png"))
            plt.close()
        #analyze the element prediction accuracy
        for atomNo in  elementsGT.keys():
            elementsCorrect = np.around(elementsGT[atomNo]).astype(int) == np.around(elementsPred[atomNo]).astype(int)
            print("element ", atomNo, " prediction accuracy: {:.2f}%".format(np.sum(elementsCorrect)/len(elementsCorrect)*100))
    except Exception as e:
        print(f"Error in {file}: {e}")
        continue
    

