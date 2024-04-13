import matplotlib.pyplot as plt
import csv
import numpy as np
import os
from rowsIndexToHeader import headerToRowsIndex
from datasets import ptychographicData

for file in os.listdir(os.path.join(os.getcwd(), "testDataEval")):
    if "results" not in file or ".csv" != file[-4:]:
        continue
    
    counter = 0
    counterCorrect = 0
    counterZeroGroundThruths = 0
    correctZeroPredictions = 0
    maxValueCorrespondsToAtom = 0
    wrongZeroPredictions = 0
    numberOfCorrectOnes = 0
    numberOfIncorrectOnes = 0

    with open(os.path.join("testDataEval", file)) as results:
        table = csv.reader(results, delimiter= ",")
        for cnt, row in enumerate(table):
            if cnt == 0 and (not "pixel" in row[0]):
                # print("Skipping file because of wrong header")
                # print("header", row)
                # print("file", file)
                break

            if len(row) %2 != 0:
                print("Skipping file because of odd number of elements in row. Should be even")
                print("row",row)
                print("file",file)
                break 

            if cnt == 0:
                continue
            
            arrayOfPixelTile = np.array(row).astype(float)

            if np.any(arrayOfPixelTile[:len(arrayOfPixelTile)//2]>0.5):
                indecesOfMaxValues = np.where(arrayOfPixelTile[:len(arrayOfPixelTile)//2] >0.5) 
                indecesOfMaxValues = np.ravel(indecesOfMaxValues)
                for indexOfMaxValues in indecesOfMaxValues:
                    if arrayOfPixelTile[len(arrayOfPixelTile)//2:][indexOfMaxValues] > 0.5:
                        numberOfCorrectOnes += 1
                    else:
                        numberOfIncorrectOnes += 1
            
            #the first half of the row is the pixel tile, the second half is the ground truth
            #check if the first half matches the second half
            if np.array_equal(np.around(arrayOfPixelTile[:len(arrayOfPixelTile)//2]).astype(int), np.around(arrayOfPixelTile[len(arrayOfPixelTile)//2:]).astype(int)):
                counterCorrect += 1
            if not np.any(np.around(arrayOfPixelTile[len(arrayOfPixelTile)//2:]).astype(int)):
                counterZeroGroundThruths += 1
            else:
                if np.any(np.around(arrayOfPixelTile[:len(arrayOfPixelTile)//2]).astype(int)):
                    wrongZeroPredictions += 1
                onesInGT = np.around(arrayOfPixelTile[len(arrayOfPixelTile)//2:]).astype(int) == 1
                maxValuesInPrediction = np.around(arrayOfPixelTile[:len(arrayOfPixelTile)//2]).astype(int) == max(np.around(arrayOfPixelTile[:len(arrayOfPixelTile)//2]).astype(int))
                if np.array_equal(maxValuesInPrediction, onesInGT):
                    maxValueCorrespondsToAtom += 1
                    # print(onesInGT)
                    # print(maxValuesInPrediction)
                    # exit()
            if not np.any(np.around(arrayOfPixelTile[len(arrayOfPixelTile)//2:]).astype(int)) and not np.any(np.around(arrayOfPixelTile[:len(arrayOfPixelTile)//2]).astype(int)):
                correctZeroPredictions += 1
            counter += 1
    if counter == 0:
        continue
    print(f"File {file} has {counterCorrect} correct pixel tiles out of {counter} total pixel tiles. That is {counterCorrect/counter*100:.2f}% correct.")
    print(f"Of all tiles, {counterZeroGroundThruths} have no atoms in them. That is {counterZeroGroundThruths/counter*100:.2f}% of all tiles.")
    print(f"Of all tiles with an atom in the ground truth {maxValueCorrespondsToAtom} have the highest prediction at the correct value. That is {maxValueCorrespondsToAtom/(counter-counterZeroGroundThruths)*100:.2f}% of all tiles with an atom in the ground truth.")
    print(f"There were {correctZeroPredictions} correct predictions of tiles with no atoms in them. That is {correctZeroPredictions/counterZeroGroundThruths*100:.2f}% of all tiles with no atoms in them. ")
    print(f"There were {wrongZeroPredictions} wrong all zero predictions even though the tiles had atoms in them. That is {wrongZeroPredictions/(counter - counterZeroGroundThruths)*100:.2f}% of all tiles with atoms in them.")
    try:
        print(f"The values over 0.5 in the prediction corresponds to an atom (== 1) in the ground truth in {numberOfCorrectOnes} cases. It does not in {numberOfIncorrectOnes} of cases. If you picked them you'd be correct in {100*numberOfCorrectOnes/(numberOfIncorrectOnes+numberOfCorrectOnes):.2f}% of times \n")
    except ZeroDivisionError:
        print("numberOfIncorrectOnes+numberOfCorrectOnes ist zero??\n")