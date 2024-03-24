import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')

###Specify file Dimension and path
Dimension = 400
path = 'PotentialReal200.bin'
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
