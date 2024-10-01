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
        
        csvFile = pandas.read_csv(os.path.join("testDataEval", file))
        if not "pixelx1y1" in csvFile.columns:
            continue
        print(f"\nRun {file}")

        modelName = "_".join(file.split("_")[1:])
        modelName = ".".join(modelName.split(".")[:-1])
        
        difs = []
        for row in tqdm(csvFile.iterrows(), desc = f"Going trough rows in file {file}", total = len(csvFile), leave = False):
            rowNum = row[1].to_numpy()
            if len(rowNum) == 0:
                continue
            rowGTs = np.array(rowNum[len(rowNum)//2:]).reshape((75,75))
            rowPreds = np.array(rowNum[:len(rowNum)//2]).reshape((75,75))*100 #(was scaled down by 100 before)
            plt.imsave(os.path.join("testDataEval", file + f"row{row[1].name}_pred.png"), rowPreds)
            plt.imsave(os.path.join("testDataEval", file + f"row{row[1].name}_GT.png"), rowGTs)
            dif = np.abs(rowGTs-rowPreds)
            difs.append(dif)
        print(f"Mean difference per Pixel: {np.mean(difs)}")

    except Exception as e:
        print(f"Error in {file}: {e}")
        continue
    

