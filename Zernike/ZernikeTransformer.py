import numpy as np
import os
import csv
from tqdm import tqdm
from mahotas.features import zernike_moments
import matplotlib.pyplot as plt
try:
    from ZernikePolynomials import Zernike
except ModuleNotFoundError:
    from Zernike.ZernikePolynomials import Zernike

def zernikeTransformation(pathToZernikeFolder = os.getcwd(), radius = 15, noOfMoments = 40, leave = True):
    oldDir = os.getcwd() 
    os.chdir(pathToZernikeFolder)
    ZernikeObject = None
    for testOrTrain in ["test", "train"]:
        imgPath = os.path.join("..","FullPixelGridML",f"measurements_{testOrTrain}")
        with open(os.path.join(f'measurements_{testOrTrain}','labels.csv'), 'w+', newline='') as labelsZernike, open(os.path.join(imgPath, 'labels.csv'), 'r', newline='') as labelsFullPixelGrid:
            Writer = csv.writer(labelsZernike, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            Reader = csv.reader(labelsFullPixelGrid, delimiter=',', quotechar='|')
            for row in Reader: #writes the title row
                Writer.writerow(row)
                break

            for row in tqdm(Reader, desc = f"Converting {testOrTrain} Data", leave =leave, total = len(os.listdir(imgPath))-1):
                if "element" in row:
                    continue
                fileName = row[0]
                if ".npy" not in fileName:
                    raise Exception(fileName + " is not a valid filename")
                image = np.load(os.path.join(imgPath, fileName))
                if ZernikeObject is None:
                    radius = radius or int(len(image)/2)
                    ZernikeObject = Zernike(radius, image.shape[-1], noOfMoments)
                # exit()
                assert(len(np.shape(image)) in [2,3])
                if len(np.shape(image)) == 3:
                    moments = []
                    for im in image:
                        moments.append(ZernikeObject.calculateZernikeWeights(im)*1e3)
                    moments = np.array(moments).flatten()
                else:
                    moments = ZernikeObject.calculateZernikeWeights(image)* 1e3 #scaled up so it's more useful
                # moments = zernike_moments(image, radius, 40) #modified zernike_moments so it doesn't output the abs values, otherwise directional analytics are not possible

                np.save(os.path.join(f"measurements_{testOrTrain}", fileName), moments)
                
                Writer.writerow(row)    
    os.chdir(oldDir)

if __name__ == "__main__":
    zernikeTransformation()