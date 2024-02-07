import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import os
import h5py

class Zernike(object):
    def __init__(self,maxR, pixelsDim, numberOfOSAANSIMoments:int):
        self.maxR = maxR + 0.5
        self.center = np.array([pixelsDim/2 - 1,pixelsDim/2 - 1])
        self.rGrid = np.zeros((pixelsDim,pixelsDim))
        self.angleGrid = np.copy(self.rGrid)
        self.numberOfOSAANSIMoments = numberOfOSAANSIMoments
        for x in range(pixelsDim):
            for y in range(pixelsDim):
                vectorFromCenter = x - self.center[0] + (y - self.center[1])*1j
                r = np.abs(vectorFromCenter)
                phi = np.angle(vectorFromCenter)
                self.rGrid[x,y] = r
                self.angleGrid[x,y] = (phi + 3*np.pi/2)%(2*np.pi)
        self.basis = self.computeZernikeBasis()
        # for cnt, basis in enumerate(self.basis):
        #     plt.imsave(f"{cnt}.png", basis)
        self.dx = 1/self.maxR
        self.dy = self.dx

    def zernikeTransform(self, fileName, images, zernikeTotalImages):
        shapeOfZernikeMoments = list(np.shape(images))[:-2]
        shapeOfZernikeMoments[-1] = len(self.basis) * shapeOfZernikeMoments[-1]
        momentsAllCoords = np.zeros(shapeOfZernikeMoments)
        for xCNT, lineOfGroupsOfPatterns in enumerate(images): #X Coord
            for yCNT, groupOfPatterns in enumerate(lineOfGroupsOfPatterns): #Y Coord
                moments = []
                for im in groupOfPatterns:
                    
                    if not np.any(im):
                        moments.append(np.zeros(len(self.basis)))
                    else:
                        moments.append(self.calculateZernikeWeights(im)*1e3) #scaled up so it's more useful
                moments = np.array(moments).flatten()
                momentsAllCoords[xCNT,yCNT] = moments

        
        zernikeTotalImages.create_dataset(fileName, data = momentsAllCoords, compression="gzip")
    def calculateZernikeWeights(self, image):
        #normFactor = np.pi #not used otherwise the weights are very small
        weights = np.sum(self.basis * image[None,:] * self.dx * self.dy , axis = (1,2))
        return weights

    def computeZernikeBasis(self):
        basis = []
        
        for n in range(self.numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                radialPart = self.ZernikePolynomialRadial(n,np.abs(m))
                angularPart = self.ZernikePolynomialAngular(m)
                basis.append(radialPart*angularPart) # pixel representation of basis indexed by OSAIndex

        return np.array(basis)

    def ZernikePolynomialAngular(self, m:int, angularVector = None) -> np.ndarray:
        if angularVector is None:
            angularVector = self.angleGrid
        if m < 0:
            angularFunc = np.sin 
        else:
            angularFunc = np.cos

        return angularFunc(abs(m) * angularVector)
        
    def ZernikePolynomialRadial(self, n:int, m:int, rValuesInput = None) -> np.ndarray:
        assert(n>=m>=0)
        assert((n-m)%2 == 0)
        k = np.arange((n-m)/2 + 1)
        summationTerms = ((-1)**k)*factorial(n-k)/(factorial(k)*factorial((n+m)/2 - k)*factorial((n-m)/2 - k)) #taken from wikipedia
        rValuesInput = rValuesInput if rValuesInput is not None else self.rGrid
        rValues = rValuesInput.flatten()
        rMatrix = np.power(rValues[:,None]/self.maxR,n-2*k)
        rMatrix[rValues > self.maxR, :] = 0 #set everything outside a specified radius to zero
        rVector = rMatrix @ summationTerms
        normalizationFactor = np.sqrt(2*(n+1)) if m else np.sqrt(n+1) #per https://iopscience.iop.org/article/10.1088/2040-8986/ac9e08
        rVector = rVector.reshape(np.shape(rValuesInput)) * normalizationFactor
        assert(not (rVector[rValuesInput > self.maxR]).any())
        return rVector
    
    def OSAANSIIndexToMNIndex(self, OSAANSIIndex:int):
        n = int((np.sqrt(8 * OSAANSIIndex + 1) - 1) / 2)
        m = 2 * OSAANSIIndex - n * (n + 2)
        return m,n
    