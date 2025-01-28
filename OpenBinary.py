import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')

import os
import re

def find_highest_numbered_file(directory, pattern=r"PotentialReal(\d+)\.bin"):
    highest_number = -1
    highest_file = None

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            number = int(match.group(1))  # Extract the number
            if number > highest_number:
                highest_number = number
                highest_file = filename

    return highest_file



###Specify file Dimension and path
Dimension = 400
path = find_highest_numbered_file(".")
print("Reading file: ", path)
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
plt.title("Reconstruction at iteration 5", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 16)
plt.savefig("".join(path.split(".")[:-1]) + ".png" )
