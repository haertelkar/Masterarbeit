import matplotlib.pyplot as plt
import csv
import numpy as np

distance = []
distancePredictionDelta = []
elements = []
zAtomsDistance = []

with open("resultsTest_cnn0.001.m.csv") as results:
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
plt.savefig("distanceToAccuracyCNN.png")
plt.show()   

""" -------------------------------- """
distance = []
distancePredictionDelta = []
elements = []
zAtomsDistance = []

with open("resultsTest_unet0.001.m.csv") as results:
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
plt.savefig("distanceToAccuracyUNN.png")
plt.show()   

""" -------------------------------- """
distance = []
distancePredictionDelta = []
elements = []
distancePred = []
zAtomsDistance = []

with open("resultsTest_znn0.001.m.csv") as results:
    table = csv.reader(results)
    for row in table: 
        if "element" in row:
            continue
        elementNN,xAtomRelNN,yAtomRelNN, zAtomsNN,element,xAtomRel,yAtomRel, zAtoms = row
        distancePredictionDelta.append(np.sqrt((float(xAtomRelNN) - float(xAtomRel))**2 + (float(yAtomRelNN) - float(yAtomRel))**2))
        distance.append((float(xAtomRel)**2 + float(yAtomRel)**2)**(1/2))
        distancePred.append((float(xAtomRelNN)**2 + float(yAtomRelNN)**2)**(1/2))
        elements.append(int(np.around(float(elementNN))) == int(float(element)))
        zAtomsDistance.append(np.abs(float(zAtomsNN)-float(zAtoms)))


distancePredictionDelta = np.array(distancePredictionDelta) / max(distancePredictionDelta)
print("element prediction accuracy: {:.2f}%".format(np.sum(elements)/len(elements)*100))
print("average error in predicting thickness: {:.2f}.".format(np.sum(zAtomsDistance)/len(zAtomsDistance)))

plt.scatter(distance, distancePredictionDelta)
plt.xlabel("Distance between atom and scan position")
plt.ylabel("Delta between distance prediction and actual distance") 
plt.savefig("distanceToAccuracy.png")
plt.show()   

plt.scatter(distance, np.array(distancePred))
plt.xlabel("Actual distance between atom and scan position")
plt.ylabel("Predicted distance") 
plt.savefig("distanceToDistanceZNN.png")
plt.show()   
   
