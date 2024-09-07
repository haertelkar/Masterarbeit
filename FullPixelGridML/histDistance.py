import csv
import math
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
filename = 'measurements_train/labels.csv'
x_values = []
y_values = []

with open(filename, 'r') as file:
    reader = csv.reader(file)

    print(next(reader))  # Skip the header row
    for row in reader:
        x_values.append(float(row[5]))
        y_values.append(float(row[9]))

x_values = np.array(x_values)#-10*0.2/2
y_values = np.array(y_values)#-10*0.2/2

# Calculate the distances
distances = [math.sqrt(x**2 + y**2) for x, y in zip(x_values, y_values)]
print("Median distance: ", np.median(distances))
print("Median distance: ", np.average(distances))

heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=1000)
xIndexZero, yIndexZero = np.digitize([0], xedges)[0], np.digitize([0], yedges)[0]
xIndexMax, yIndexMax = np.unravel_index(heatmap.argmax(), heatmap.shape)
print(f"Zero index: {xIndexZero}, {yIndexZero}")
print(f"Max index: {xIndexMax}, {yIndexMax}")
distanceBetweenCoordsX = xedges[1]-xedges[0]
distanceBetweenCoordsY = yedges[1]-yedges[0]
print(f"x-Distance between zero and max: {distanceBetweenCoordsX*(xIndexZero-xIndexMax)}") 
print(f"y-Distance between zero and max: {distanceBetweenCoordsY*(yIndexZero-yIndexMax)}")
print(f"Distance between zero and max: {math.sqrt((distanceBetweenCoordsX*(xIndexZero-xIndexMax))**2 + (distanceBetweenCoordsY*(yIndexZero-yIndexMax))**2)}")

# print(f"xMax = {(xedges[1]-xedges[0])/1000*np.where(heatmap==heatmap.max())-(xedges[1]-xedges[0])/2}")
# print(f"yMax = {(yedges[1]-yedges[0])/1000*np.argmax(heatmap, axis = 1)-(yedges[1]-yedges[0])/2}")
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
logHeatmap = np.log(heatmap.T, where=heatmap.T != 0)
logHeatmap[heatmap.T == 0] = np.log(1) - 1 #to make zero the lowest value
plt.clf()
plt.imshow(logHeatmap, extent=extent, origin='lower')
plt.plot(0, 0, marker='v', color="white")
plt.savefig("xyHeatmap.png")
plt.close()
# Plot the histogram
plt.hist(distances, bins=1000)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Histogram of Distances')
plt.savefig("distance_histogram.png")