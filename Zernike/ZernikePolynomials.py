import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import os
import h5py
import numexpr as ne





class Zernike(object):
    def __init__(self,maxR, numberOfOSAANSIMoments:int):
        self.maxR = maxR + 0.5
        self.numberOfOSAANSIMoments = numberOfOSAANSIMoments
        self.dx = 1/self.maxR
        self.dy = self.dx
        self.indexXNonZero = None
        self.indexYNonZero = None
        self.dimToBasis = {
        }
        self.resultVectorLength = 0 

        for n in range(self.numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                self.resultVectorLength += 1

    # def zernikeTransformMultiImages(self, fileName, images, zernikeTotalImages, shapeOfMomentsAllCoords = None):
    #     if shapeOfMomentsAllCoords is None: 
    #         shapeOfZernikeMoments = list(np.shape(images))[:-2]
    #         shapeOfZernikeMoments[-1] = self.resultVectorLength * shapeOfZernikeMoments[-1]
    #         momentsAllCoords = np.zeros(shapeOfZernikeMoments)
    #     else:
    #         shapeOfMomentsAllCoords[-1] = self.resultVectorLength * shapeOfMomentsAllCoords[-1]
    #         momentsAllCoords = np.zeros(shapeOfMomentsAllCoords)
    #     for xCNT, lineOfGroupsOfPatterns in enumerate(images): #X Coord
    #         for yCNT, groupOfPatterns in enumerate(lineOfGroupsOfPatterns): #Y Coord
    #             moments = []
    #             for im in groupOfPatterns:
    #                 dim = np.shape(im)[0]
    #                 if self.dimToBasis.get(dim) is None:
    #                     basisObject = self.basisObject(self, dim)
    #                     self.dimToBasis[dim] = basisObject.basis
    #                 basis = self.dimToBasis[dim]
    #                 if not np.any(im):
    #                     #most diffraction patterns are left empty, so this is a good optimization
    #                     moments.append(np.zeros(self.resultVectorLength))
    #                 else:
    #                     moments.append(self.calculateZernikeWeights(basis, im)*1e3) #scaled up so it's more useful
    #             moments = np.array(moments).flatten()
    #             momentsAllCoords[xCNT,yCNT] = moments

        
    #     if fileName is not None:
    #         zernikeTotalImages.create_dataset(fileName, data = momentsAllCoords, compression="gzip")
    #     else:
    #         return momentsAllCoords
    
    def calculateZernikeWeightsOnWholeGroup(self, basis, groupOfPatterns):
        return np.sum(basis[None,:,self.indexXNonZero,self.indexYNonZero]* groupOfPatterns[:,None,self.indexXNonZero,self.indexYNonZero], axis = (2,3)).flatten()

    def calculateZernikeWeights(self, basis, image):
        #normFactor = np.pi #not used otherwise the weights are very small
        #normally the weights are normalized by the area of the image (self.dx * self.dy), but this is not necessary for our purposes
        return np.sum(basis[:,self.indexXNonZero,self.indexYNonZero]* image[None,self.indexXNonZero,self.indexYNonZero], axis = (1,2)).flatten()

    
    def zernikeTransform(self, fileName, groupOfPatterns, zernikeTotalImages, shapeOfMomentsAllCoords = None):
        assert(len(np.shape(groupOfPatterns)) == 3)
        moments = []
        for im in groupOfPatterns:
            dim = np.shape(im)[-1]
            if self.dimToBasis.get(dim) is None:
                basisObject = self.basisObject(self, dim)
                self.dimToBasis[dim] = basisObject.basis.copy()
            basis = self.dimToBasis[dim]
            if not np.any(im):
                #most diffraction patterns are left empty, so this is a good optimization
                moments.append(np.zeros(self.resultVectorLength))
            else:
                moments.append(self.calculateZernikeWeights(basis, im)) #before: scaled up with 10e3 so it's more useful
        moments = np.array(moments).flatten()


        
        if fileName is not None:
            zernikeTotalImages.create_dataset(fileName, data = moments, compression="gzip")
        else:
            return moments
        
           
    def zernikeTransformVectorized(self, fileName, groupOfPatterns, zernikeTotalImages, shapeOfMomentsAllCoords = None):
        #Slower for some weird reason
        assert(len(np.shape(groupOfPatterns)) == 3)
        dim = np.shape(groupOfPatterns)[-1]
        if self.dimToBasis.get(dim) is None:
            basisObject = self.basisObject(self, dim)
            self.dimToBasis[dim] = basisObject.basis.copy()
        basis = self.dimToBasis[dim]
        moments = self.calculateZernikeWeightsOnWholeGroup(basis, groupOfPatterns)

        if fileName is not None:
            zernikeTotalImages.create_dataset(fileName, data = moments, compression="gzip")
        else:
            return moments
    



    
    class basisObject:
        def __init__(self, ZernikeObject, pixelsDim):
            self.ZernikeObject = ZernikeObject
            self.pixelsDim = pixelsDim
            self.center = np.array([pixelsDim/2 - 1,pixelsDim/2 - 1])
            self.rGrid = np.zeros((pixelsDim,pixelsDim))
            self.angleGrid = np.copy(self.rGrid)
            self.ZernikeObject.indexXNonZero = slice(min(int(pixelsDim//2 - self.ZernikeObject.maxR),0), max(int(pixelsDim//2 + self.ZernikeObject.maxR), int(pixelsDim)))
            self.ZernikeObject.indexYNonZero = self.ZernikeObject.indexXNonZero

            for x in range(pixelsDim):
                for y in range(pixelsDim):
                    vectorFromCenter = x - self.center[0] + (y - self.center[1])*1j
                    r = np.abs(vectorFromCenter)
                    phi = np.angle(vectorFromCenter)
                    self.rGrid[x,y] = r
                    self.angleGrid[x,y] = (phi + 3*np.pi/2)%(2*np.pi)

            self.basis = self.computeZernikeBasis()
            # for cnt, order in enumerate(self.basis):
            #     plt.imsave(f"{cnt}.png", basis)
        
        def computeZernikeBasis(self):
            basis = []
            
            for n in range(self.ZernikeObject.numberOfOSAANSIMoments + 1):
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
            rValues = rValuesInput.copy()
            rValues = rValues.flatten()
            rMatrix = np.power(rValues[:,None]/self.ZernikeObject.maxR,n-2*k)
            rMatrix[rValues > self.ZernikeObject.maxR, :] = 0 #set everything outside a specified radius to zero
            rVector = rMatrix @ summationTerms
            normalizationFactor = np.sqrt(2*(n+1)) if m else np.sqrt(n+1) #per https://iopscience.iop.org/article/10.1088/2040-8986/ac9e08
            rVector = rVector.reshape(np.shape(rValuesInput)) * normalizationFactor
            assert(not (rVector[rValuesInput > self.ZernikeObject.maxR]).any())
            return rVector
        
        def OSAANSIIndexToMNIndex(self, OSAANSIIndex:int):
            n = int((np.sqrt(8 * OSAANSIIndex + 1) - 1) / 2)
            m = 2 * OSAANSIIndex - n * (n + 2)
            return m,n
    