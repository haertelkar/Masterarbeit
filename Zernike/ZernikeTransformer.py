import mahotas
import numpy as np
import os
import csv
from tqdm import tqdm
import pandas as pd

class ZernikeMoments:
    #https://pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/
	def __init__(self, radius):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius
	def describe(self, image):
		# return the Zernike moments for the image
		return mahotas.features.zernike_moments(image, self.radius, degree = 8)

imgPathTest = "..\\FullPixelGridML\\measurements_test"

with open('measurements_test\\labels.csv', 'w+', newline='') as labelsZernike, open(imgPathTest + '\\labels.csv', 'r', newline='') as labelsFullPixelGrid:
    Writer = csv.writer(labelsZernike, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    Reader = csv.reader(labelsFullPixelGrid, delimiter=',', quotechar='|')
    moments = None
    for firstRow in Reader:
        Writer.writerow(["fileName","element","xAtomRel","yAtomRel","zAtoms"])
        break

    for row in tqdm(Reader, desc = "Converting Test Data"):
        if "element" in row:
            continue
        fileName, element, xAtomRel, xAtomShift, yAtomRel, yAtomShift, zAtoms = row
        if ".npy" not in fileName:
            raise Exception(fileName + " is not a valid filename")
        image = np.load(os.path.join(imgPathTest, fileName))
        if moments is None:
            radius = 7#int(len(image)/2) - 1 
            desc = ZernikeMoments(radius)
        moments = desc.describe(image)          
        np.save("measurements_test\\" + fileName, moments)
        
        #Writer.writerow([fileName] + [str(difParams) for difParams in [element, xAtomRel, xAtomShift, yAtomRel, yAtomShift]])    
        Writer.writerow([fileName] + [str(difParams) for difParams in [element, xAtomRel, yAtomRel, zAtoms]])

imgPathTrain = "..\\FullPixelGridML\\measurements_train"

with open('measurements_train\\labels.csv', 'w+', newline='') as labelsZernike, open(imgPathTrain + '\\labels.csv', 'r', newline='') as labelsFullPixelGrid:
    Writer = csv.writer(labelsZernike, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    Reader = csv.reader(labelsFullPixelGrid, delimiter=',', quotechar='|')
    
    for firstRow in Reader:
        Writer.writerow(["fileName","element","xAtomRel","yAtomRel","zAtoms"])
        break

    for row in tqdm(Reader, desc = "Converting Test Data"):
        if "element" in row:
            continue
        fileName, element, xAtomRel, xAtomShift, yAtomRel, yAtomShift, zAtoms = row
        if ".npy" not in fileName:
            raise Exception(fileName + " is not a valid filename")
        image = np.load(os.path.join(imgPathTrain, fileName))
        moments = desc.describe(image)                       
        np.save("measurements_train\\" + fileName, moments)
        #Writer.writerow([fileName] + [str(difParams) for difParams in [element, xAtomRel, xAtomShift, yAtomRel, yAtomShift]])   
        Writer.writerow([fileName] + [str(difParams) for difParams in [element, xAtomRel, yAtomRel, zAtoms]])