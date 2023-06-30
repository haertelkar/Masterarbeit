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
    elements = []
    zAtomsDistance = []
    skipFile = False

    with open(file) as results:
        table = csv.reader(results, delimiter= ",")
        for row in table: 
            if "element" in row:
                continue
            try:
                elementNN,xAtomRelNN,xAtomShiftNN,yAtomRelNN,yAtomShiftNN, zAtomsNN, element,xAtomRel,xAtomShift,yAtomRel,yAtomShift, zAtoms = row
            except ValueError:
                if len(row) > 2:
                    elementNN,xAtomRelNN,yAtomRelNN,zAtomsNN, element,xAtomRel,yAtomRel,zAtoms = row
                else:
                    skipFile = True
                    break           
            distancePredictionDelta.append(np.sqrt((float(xAtomRelNN) - float(xAtomRel))**2 + (float(yAtomRelNN) - float(yAtomRel))**2))
            
            distance.append((float(xAtomRel)**2 + float(yAtomRel)**2)**(1/2))
            elements.append(int(np.around(float(elementNN))) == int(float(element)))
            zAtomsDistance.append(np.abs(float(zAtomsNN)-float(zAtoms)))
    if skipFile:
        continue
    try:
        distancePredictionDelta = np.array(distancePredictionDelta) / (max(distancePredictionDelta) + 1e-5)
    except Exception as e:
        print(e)
        print("in file ", file)
        raise Exception
    modelName = "_".join(file.split("_")[1:])
    modelName = ".".join(modelName.split(".")[:-1])
    print(modelName)

    errors.append([modelName, np.sum(zAtomsDistance)/len(zAtomsDistance), np.sum(elements)/len(elements)*100, np.sum(distancePredictionDelta)**2])
    print("average error in predicting thickness: {:.2f}.".format(np.sum(zAtomsDistance)/len(zAtomsDistance)))
    print("element prediction accuracy: {:.2f}%".format(np.sum(elements)/len(elements)*100))
    print(f"MSE distance {np.sum(distancePredictionDelta)**2:2e}")
    plt.scatter(distance, distancePredictionDelta)
    plt.xlabel("Distance between atom and scan position")
    plt.ylabel("Delta between distance prediction and actual distance") 
    plt.savefig(file + "distanceToAccuracyCNN.pdf")
    plt.close()
    print()
    #plt.show()   


with open("errors.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter= ";")
     for row in errors:
        wr.writerow(row)
