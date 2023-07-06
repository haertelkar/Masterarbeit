import matplotlib.pyplot as plt
import csv
import numpy as np
import os


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
        for row in table:
            if len(row) != 18:
                skipFile = True
                break 
            if "element1" in row:
                continue
            elementsNN = np.array([float(i) for i in row[:3]])
            xAtomRelsNN = np.array([float(i) for i in row[3:6]])
            yAtomRelsNN = np.array([float(i) for i in row[6:9]])
            elements = np.array([float(i) for i in row[9:12]])
            xAtomRels = np.array([float(i) for i in row[12:15]])
            yAtomRels = np.array([float(i) for i in row[15:18]])
            distancePredictionDelta.append(np.sqrt((xAtomRelsNN - xAtomRels)**2 + (yAtomRelsNN - yAtomRels)**2))
            distance.append(np.sqrt((xAtomRels)**2 + (yAtomRels)**2))
            elementsCorrect.append(np.around(elements).astype(int) == np.around(elementsNN).astype(int))
    if skipFile:
        continue
    try:
        distancePredictionDelta = np.array(distancePredictionDelta)
        distancePredictionDelta = np.transpose(distancePredictionDelta)
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
    for e in np.array(elementsCorrect).transpose():
        print("element prediction accuracy: {:.2f}%".format(np.sum(e)/len(e)*100))
    #print(f"MSE distance {np.sum(distancePredictionDelta,axis=0)**2:2e}")
    for cnt, ds in enumerate(zip(distance,distancePredictionDelta)):
        d,dDelta = ds
        plt.scatter(d, dDelta)
        plt.xlabel(f"Distance between atom nr.{cnt} and scan position")
        plt.ylabel("Delta between distance prediction and actual distance") 
        plt.savefig(file + f"distanceToAccuracyCNN_atom{cnt}.pdf")
        plt.close()
    print()
    #plt.show()   


# with open("errors.csv", 'w', newline='') as myfile:
#      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter= ";")
#      for row in errors:
#         wr.writerow(row)
