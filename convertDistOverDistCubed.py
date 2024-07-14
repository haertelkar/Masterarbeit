import pandas as pd
import numpy as np
import csv

csvFile = "AppendedColumn_results_ZernikeNormal_0807_DistOverCubedDistOnlyDist.csv"
csvFileFullPath = "testDataEval/"+csvFile

#change the first row in csvFile to "element1", "xAtomRelOverDistCub1", "yAtomRelOverDistCub1", "element" "xAtomRelOverDistCub", "yAtomRelOverDistCub"
with open(csvFileFullPath, "r+") as results:
    lines = results.readlines()
    lines[0] = "element1,xAtomRelOverDistCub1,yAtomRelOverDistCub1,element,xAtomRelOverDistCub,yAtomRelOverDistCub\n"
    results.seek(0)
    results.writelines(lines)

print(f"Processing {csvFileFullPath} data")
labels = pd.read_csv(csvFileFullPath,  sep=",")
labels["xAtomRelOverDistCub1"] = labels["xAtomRelOverDistCub1"].apply(lambda x: np.sign(x)*np.sqrt(1/np.abs(x)))
labels["xAtomRelOverDistCub"] = labels["xAtomRelOverDistCub"].apply(lambda x: np.sign(x)*np.sqrt(1/np.abs(x)))
labels["yAtomRelOverDistCub1"] = labels["yAtomRelOverDistCub1"].apply(lambda x: np.sign(x)*np.sqrt(1/np.abs(x)))
labels["yAtomRelOverDistCub"] = labels["yAtomRelOverDistCub"].apply(lambda x: np.sign(x)*np.sqrt(1/np.abs(x)))
labels.to_csv("testDataEval/"+f"correctedFraction_{csvFile}", index = False)

#change the first row in csvFile to "element1", "xAtomRelOverDistCub1", "yAtomRelOverDistCub1", "element" "xAtomRelOverDistCub", "yAtomRelOverDistCub"
with open("testDataEval/"+f"correctedFraction_{csvFile}", "r+") as results:
    lines = results.readlines()
    lines[0] = "element,xAtomRel,yAtomRel,element,xAtomRel,yAtomRel\n"
    results.seek(0)
    results.writelines(lines)