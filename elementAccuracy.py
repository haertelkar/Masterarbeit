import csv

from matplotlib import pyplot as plt
import numpy as np

elementPreds = []
elementGTs = []

with open("testDataEval/results_ZernikeNormal_evaluation_ZernikeNormal_13107_onlyElemLonger_epoch=14-step=24600.csv", "r") as file:
    reader = csv.reader(file)
    firstRow = True
    for row in reader:
        if firstRow:
            firstRow = False
            continue
        elementPred, elementGT = row
        elementPreds.append(float(elementPred))
        elementGTs.append(int(float(elementGT)))

elementPreds = np.array(elementPreds)
elementGTs = np.array(elementGTs)

roundedPreds = np.around(elementPreds).astype(int)

distance = elementPreds - elementGTs
averageDistance = np.average(np.abs(distance))
meanDistance = np.mean(np.abs(distance))

heatmap, xedges, yedges = np.histogram2d(elementPreds, elementGTs, bins=(max(roundedPreds)-min(roundedPreds), max(elementGTs)-min(elementGTs)))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
logHeatmap = np.log(heatmap.T, where=heatmap.T != 0)
logHeatmap[heatmap.T == 0] = np.log(1) - 1 #to make zero the lowest value
plt.clf()
plt.imshow(logHeatmap, extent=extent, origin='lower')
plt.title("Element Accuracy Heatmap Logartihmic")
plt.xlabel("Predicted Element")
plt.ylabel("Ground Truth Element")
plt.savefig("elementAccuracyHeatmap.png")
plt.close()

plt.hist(distance, bins=1000)
plt.title("Element Accuracy Histogram")
plt.xlabel("Distance between predicted and ground truth element")
plt.savefig("elementAccuracyHistogram.png")
plt.close()

print(f"average np.abs(distance): {averageDistance}")
print(f"mean np.abs(distance): {meanDistance}")