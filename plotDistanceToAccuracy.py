import matplotlib.pyplot as plt
import csv
import numpy as np
import os

distance = []
distancePredictionDelta = []
elements = []
zAtomsDistance = []

for file in os.listdir(os.getcwd()):
    if not ("results" in file and ".csv" in file):
        continue
    with open(file) as results:
        table = csv.reader(results)
        for row in table: 
            if "element" in row:
                continue
            elementNN,xAtomRelNN,xAtomShiftNN,yAtomRelNN,yAtomShiftNN, zAtomsNN, element,xAtomRel,xAtomShift,yAtomRel,yAtomShift, zAtoms = row
            distancePredictionDelta.append(np.sqrt((float(xAtomRelNN) - float(xAtomRel))**2 + (float(yAtomRelNN) - float(yAtomRel))**2))
            distance.append((float(xAtomRel)**2 + float(yAtomRel)**2)**(1/2))
            elements.append(int(np.around(float(elementNN))) == int(float(element)))
            zAtomsDistance.append(np.abs(float(zAtomsNN)-float(zAtoms)))

    distancePredictionDelta = np.array(distancePredictionDelta) / max(distancePredictionDelta)

    print("average error in predicting thickness: {:.2f}.".format(np.sum(zAtomsDistance)/len(zAtomsDistance)))
    print("element prediction accuracy: {:.2f}%".format(np.sum(elements)/len(elements)*100))

    plt.scatter(distance, distancePredictionDelta)
    plt.xlabel("Distance between atom and scan position")
    plt.ylabel("Delta between distance prediction and actual distance") 
    plt.savefig(file + "distanceToAccuracyCNN.png")
    #plt.show()   

   
