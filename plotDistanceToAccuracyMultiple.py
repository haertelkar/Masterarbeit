import matplotlib.pyplot as plt
import csv
import numpy as np
import os
from rowsIndexToHeader import headerToRowsIndex

def raiseExcep(header, index):
    if type(index) is int:
        return index
    else:
        raise Exception(f"header '{header}' unknown")
    
def returnFullRowContent(fullRowIndices, fullRow, startInd, endInd):
    returnlist = []
    for i in range(startInd,endInd):
        if i in fullRowIndices:
            returnlist.append(fullRow[i])
    return returnlist

errors = [["modelName", "average error in predicting thickness", "element prediction accuracy", "MSE distance"]]

for file in os.listdir(os.getcwd()):
    if "results" not in file or ".csv" != file[-4:]:
        continue

    distance = []
    distancePredictionDelta = []
    elementsCorrect = []
    zAtomsDistance = []
    skipFile = False

    with open(file) as results:
        table = csv.reader(results, delimiter= ",")
        fullRowIndices = []
        fullRow = np.zeros(18)
        knownHeaders = []
        for row in table:
            if len(row) %2 != 0:
                skipFile = True
                break 
            if fullRowIndices == []: #only in the header row
                for header in row:
                    if header not in knownHeaders:
                        fullRowIndices.append(raiseExcep(header,headerToRowsIndex.get(header)) - 1)
                        knownHeaders.append(header)
                    else:
                        fullRowIndices.append(headerToRowsIndex.get(header) -1 + 9)
                continue
            try:
                fullRow[fullRowIndices] = row
            except ValueError as e:
                print(row)
                print(file)
                raise Exception(e)

            elementsNN = np.array([float(i) for i in returnFullRowContent(fullRowIndices, fullRow,0,3)])
            xAtomRelsNN = np.array([float(i) for i in returnFullRowContent(fullRowIndices, fullRow,3,6)])
            yAtomRelsNN = np.array([float(i) for i in returnFullRowContent(fullRowIndices, fullRow,6,9)])
            elements = np.array([float(i) for i in returnFullRowContent(fullRowIndices, fullRow,9,12)])
            xAtomRels = np.array([float(i) for i in returnFullRowContent(fullRowIndices, fullRow,12,15)])
            yAtomRels = np.array([float(i) for i in returnFullRowContent(fullRowIndices, fullRow,15,18)])
            if xAtomRels.any() and yAtomRels.any(): distancePredictionDelta.append(np.sqrt((xAtomRelsNN - xAtomRels)**2 + (yAtomRelsNN - yAtomRels)**2))
            if xAtomRels.any() and yAtomRels.any(): distance.append(np.sqrt((xAtomRels)**2 + (yAtomRels)**2))
            if elements.any(): elementsCorrect.append(np.around(elements).astype(int) == np.around(elementsNN).astype(int))
    if skipFile:
        continue
    try:
        if distancePredictionDelta:
            distancePredictionDelta = np.array(distancePredictionDelta)
            distancePredictionDelta = np.transpose(distancePredictionDelta)
        if distance:
            distance = np.transpose(distance)
        #distancePredictionDelta = distancePredictionDelta / (np.max(distancePredictionDelta, axis = 0) + 1e-5)
    except Exception as e:
        print(e)
        print("in file ", file)
        raise Exception
    modelName = "_".join(file.split("_")[1:])
    modelName = ".".join(modelName.split(".")[:-1])
    print(modelName)
    #errors.append([modelName, np.sum(zAtomsDistance)/len(zAtomsDistance), np.sum(elements)/len(elements)*100, np.sum(distancePredictionDelta)**2]))
    if elementsCorrect:
        for e in np.array(elementsCorrect).transpose():
            print("element prediction accuracy: {:.2f}%".format(np.sum(e)/len(e)*100))
    else:
        print("No elements predictions found")
    #print(f"MSE distance {np.sum(distancePredictionDelta,axis=0)**2:2e}")
    distance = np.array(distance)
    distancePredictionDelta = np.array(distancePredictionDelta)
    if distance.any() and distancePredictionDelta.any():
        for cnt, ds in enumerate(zip(distance,distancePredictionDelta)):
            d,dDelta = ds
            plt.scatter(d, dDelta)
            plt.xlabel(f"Distance between atom nr.{cnt} and scan position")
            plt.ylabel("Delta between distance prediction and actual distance") 
            plt.savefig(file + f"distanceToAccuracyCNN_atom{cnt}.pdf")
            plt.close()
    else:
        print("No xy predictions found")
    print()
    #plt.show()   


# with open("errors.csv", 'w', newline='') as myfile:
#      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter= ";")
#      for row in errors:
#         wr.writerow(row)
