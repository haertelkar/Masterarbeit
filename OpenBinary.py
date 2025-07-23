import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')

import os
import re

def find_highest_numbered_file(directory, pattern=r"PotentialReal(\d+)\.bin"):
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

    return highest_file, highest_number


for folder in ["TotalPosROP", "TotalGridROP"] + [f"PredROP{i}" for i in range(2,10)]+ [f"PredROP{i}_rndm" for i in range(2,10)]:
    ###Specify file Dimension and path
    #Dimension = 500
    paths = [find_highest_numbered_file(folder, pattern=i) for i in [r"PotentialReal(\d+)\.bin", r"ProbeReal(\d+)\.bin"]]
    #paths = [("PotentialReal500.bin",500), ("ProbeReal500.bin",500)]
    for (path, iteration), Dimension in zip(paths, [500,341]):
        if path == "No File Found":
            print(f"No files found in {folder}. Skipping...")
            continue

        path = folder + "/" + path
        print("Reading file: ", path)
        ###Open file
        try:
            data = open(path, 'rb').read()
        except FileNotFoundError:
            print(f"File {path} not found. Skipping...")
            continue
        ###Generate 2D array
        try:
            values = struct.unpack(Dimension * Dimension * 'f', data)
        except struct.error as e:
            Dimension = int(np.sqrt(len(data) / 4))
            print(f"Error unpacking data: {e}. Adjusting Dimension to {Dimension}.")
            values = struct.unpack(Dimension * Dimension * 'f', data)
                  
        values =  np.reshape(values,(Dimension,Dimension))
        ###Plot array
        from matplotlib import pyplot as plt
        yrange=np.arange(Dimension)
        xrange=np.arange(Dimension)
        x , y=np.meshgrid(xrange,yrange)
        plt.contourf(x,y, values, 500, cmap='gray')
        ###Specify some plot settings
        plt.title(f"Reconstruction at iteration {iteration}", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize = 16)
        plt.savefig("".join(path.split(".")[:-1]) + ".png" )
        plt.close()
