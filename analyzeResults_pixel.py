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
    # try:
    if "results" not in file or ".csv" != file[-4:] or not onlyThisFile in file:
        continue
    
    csvFile = pandas.read_csv(os.path.join("testDataEval", file))
    if not "pixelx1y1" in csvFile.columns:
        continue
    print(f"\nRun {file}")

    modelName = "_".join(file.split("_")[1:])
    modelName = ".".join(modelName.split(".")[:-1])
    
    difs = []
    noOfCorrectHits = 0
    noOfIncorrectHits = 0
    gts = []
    cnt = 0
    for row in tqdm(csvFile.iterrows(), desc = f"Going trough rows in file {file}", total = len(csvFile), leave = False):
        
        rowNum = row[1].to_numpy()
        if len(rowNum) == 0:
            continue
        gridSize = int(np.sqrt(len(rowNum)//2))
        rowGTs = np.array(rowNum[len(rowNum)//2:]).reshape((gridSize ,gridSize ))
        rowPreds = np.array(rowNum[:len(rowNum)//2]).reshape((gridSize ,gridSize)) #(was scaled down by 100 before)
        gts.append(rowGTs) 
        dif = np.abs(rowGTs-rowPreds)
        difs.append(dif)
        GTatoms = np.round(rowGTs*2-1).astype(int)
        PredAtoms = np.round(rowPreds).astype(int)

        noOfCorrectHits += np.sum(np.clip(np.clip(GTatoms,0,1)*PredAtoms,0,1))
        noOfIncorrectHits -= np.sum(np.clip(np.clip(GTatoms,-1,0)-PredAtoms,-1,0))
        # print(rowPreds)
        # # exit(   )
        # print(f"np.mean(rowGTs): {np.mean(rowGTs)}")
        # print(f"np.mean(rowPreds): {np.mean(rowPreds)}")
        #plot rowPreds and rowGTs in one file with colorbars
        cnt += 1
        if cnt % 150 != 0:
            continue
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(rowPreds)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(rowGTs)
        plt.colorbar()
        plt.suptitle(f"Model: {modelName}")
        plt.savefig(os.path.join("testDataEval", file + f"row{cnt}_comparison.png"))
        plt.close()

    print(f"correct minus incorrect preds {noOfCorrectHits-noOfIncorrectHits}")
    print(f"correct preds per image {noOfCorrectHits/cnt}")
    print(f"incorrect preds per image {noOfIncorrectHits/cnt}")
    print(f"Mean difference per Pixel: {np.mean(difs)}")
    print(f"np.max(gts){np.max(gts)}")
    print(f"np.min(gts){np.min(gts)}")
    # except Exception as e:
    #     print(f"Error in {file}: {e}")
    #     continue
    

