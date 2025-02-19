import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')

import os
import re

def find_every_500th_iteration(directory, pattern=r"PotentialReal(\d+)\.bin"):
    highest_number = -1
    highest_file = "No File Found"

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            number = int(match.group(1))  # Extract the number
            if number > highest_number:
                highest_number = number
                highest_file = filename
            if number % 500 == 0:
                yield filename, number

    # return highest_file, highest_number

NameResults = input("What should the results be named? ")

newFolder = os.path.join("ROPResults",NameResults)
try:
    os.mkdir(os.path.join("ROPResults",NameResults))
except FileExistsError:
    print("Folder already exists, choose different name")
    exit()

#copy "Predictions.png" to new folder
import shutil
shutil.copy("Predictions.png", newFolder)
shutil.copy("groundTruth.png", newFolder)
shutil.copy("testStructureOriginal.png", newFolder)

for folder in ["TotalPosROP"] + [f"PredROP{i}" for i in range(2,10)] + [f"PredROP{i}_rndm" for i in range(2,10)]:
    ###Specify file Dimension and path
    Dimension = 500
    appendix = ""
    text = ""
    for filename , iteration in find_every_500th_iteration(folder):
        if "rndm" in folder:
            appendix = "_rndm"
            text = " random"
        rootPath = folder + "/" 
        path = rootPath + filename
        print("Reading file: ", path)
        fileNameWithOutExtension = "".join(filename.split(".")[:-1])
        pathWithOutExtension =  rootPath + fileNameWithOutExtension
        #open the text file "positions10.txt" and count the number of lines to find the number
        #of positions
        with open(rootPath + "Positions10.txt") as f:
            lines = f.readlines()
            numberOfPositions = len(lines)
        print("\tNumber of positions: ", numberOfPositions)
        ###Open file
        data = open(path, 'rb').read()
        ###Generate 2D array
        values = struct.unpack(Dimension * Dimension * 'f', data)
        values =  np.reshape(values,(Dimension,Dimension))
        ###Plot array
        from matplotlib import pyplot as plt
        yrange=np.arange(Dimension)
        xrange=np.arange(Dimension)
        x , y=np.meshgrid(xrange,yrange)
        plt.contourf(x,y, values, 500, cmap='gray')
        ###Specify some plot settings
        plt.title(f"Reconstruction with {numberOfPositions}{text} scan positions\n"+
                f"at iteration {iteration}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize = 16)
        try:
            os.mkdir(f"{newFolder}/NumOfPos_{numberOfPositions}{appendix}")
        except FileExistsError:
            pass
        shutil.copy(path, f"{newFolder}/NumOfPos_{numberOfPositions}{appendix}")
        plt.savefig(f"{newFolder}/NumOfPos_{numberOfPositions}{appendix}/{fileNameWithOutExtension}.png")
        plt.close()
