import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import os

def imagePath(testOrTrain):
    return os.path.join("..","FullPixelGridML",f"measurements_{testOrTrain}")

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

    def zernikeTransform(self, testOrTrain, fileName):
        try:
            image = np.load(os.path.join(imagePath(testOrTrain), fileName))
        except ValueError as e:
            raise Exception(e +f"\nError in {fileName}")
        assert(len(np.shape(image)) in [2,3])
        if len(np.shape(image)) == 3:
            moments = []
            for im in image:
                moments.append(self.calculateZernikeWeights(im)*1e3)
            moments = np.array(moments).flatten()
        else:
            moments = self.calculateZernikeWeights(image)* 1e3 #scaled up so it's more useful
        np.save(os.path.join(f"measurements_{testOrTrain}", fileName), moments)
            # moments = zernike_moments(image, radius, 40) #modified zernike_moments so it doesn't output the abs values, otherwise directional analytics are not possible

    def calculateZernikeWeights(self, image):
        #normFactor = np.pi #not used otherwise the weights are very small
        weights = np.sum(self.basis * image[None,:] * self.dx * self.dy , axis = (1,2))
        # for cnt in range(len(self.basis)):
        #     randomNumber1 = np.random.randint(0, len(self.basis))
        #     randomNumber2 = np.random.randint(0, len(self.basis))
        #     if randomNumber1 == randomNumber2:
        #         continue
        #     e = np.sum(self.basis[randomNumber1] * self.basis[randomNumber2] * self.dx * self.dy/np.pi)   
        #     print(f"n0 = {self.OSAANSIIndexToMNIndex(randomNumber1)[1]}, m0 = {self.OSAANSIIndexToMNIndex(randomNumber1)[0]}, n1 = {self.OSAANSIIndexToMNIndex(randomNumber2)[1]}, m1 = {self.OSAANSIIndexToMNIndex(randomNumber2)[0]}",e)
        #exit()
        return weights

    def computeZernikeBasis(self):
        basis = []
        # radialParts = []
        # angularParts = []
        # nOlds = []
        # ms = []
        
        for n in range(self.numberOfOSAANSIMoments + 1):
            for mShifted in range(2*n+1):
                m = mShifted - n
                if (n-m)%2 != 0:
                    continue
                #m,n = self.OSAANSIIndexToMNIndex(index)
                radialPart = self.ZernikePolynomialRadial(n,np.abs(m))
                # radialParts.append(self.ZernikePolynomialRadialSimple(n,np.abs(m), np.arange(self.maxR + 0.5).astype('float64')))
                # nOlds.append(n)
                # ms.append(m)
                angularPart = self.ZernikePolynomialAngular(m)
                basis.append(radialPart*angularPart) # pixel representation of basis indexed by OSAIndex
        # radialParts = np.array(radialParts)
        # angularParts = np.array(angularParts)
        # for cnt in range(len(radialParts)):#len(radialParts)):
        #     randomNumber1 = np.random.randint(0, len(radialParts))
        #     randomNumber2 = np.random.randint(0, len(radialParts))
        #     if not (nOlds[randomNumber2] != nOlds[randomNumber1] and ms[randomNumber1] == ms[randomNumber2]):
        #         continue
        #     # if randomNumber1 == randomNumber2:
        #     #     continue
        #     no = 10000
        #     x = np.linspace(0,1, no, endpoint= True)
        #     intParts = radialParts[randomNumber1] * radialParts[randomNumber2] * np.arange(self.maxR + 0.5)/(self.maxR)
        #     dr = 1/(self.maxR)
        #     #e = np.sum(np.interp(x,(np.arange(self.maxR + 0.5))/(self.maxR), intParts)) /no
        #     e = np.sum(intParts) * dr
        #     print(f"n0 = {nOlds[randomNumber1]}, m0 = {ms[randomNumber1]}, n1 = {nOlds[randomNumber2]}, m1 = {ms[randomNumber2]}, product = {e} ")
        #     plt.plot(np.arange(self.maxR + 0.5)/(self.maxR), radialParts[randomNumber1]* radialParts[randomNumber2])
        #     plt.savefig(f"n{nOlds[randomNumber1]}, m{ms[randomNumber1]}_radial.png")
        #     plt.close()
        # for cnt, e in enumerate(np.sum(radialParts[1:] * radialParts[:-1] * np.arange(self.maxR+ 1)/self.maxR * dr, axis = 1)):
        #     print(f"n0 = {nOlds[cnt]}, m0 = {ms[cnt]}, n1 = {nOlds[cnt + 1]}, m1 = {ms[cnt + 1]}, product = {e} ")
        # print(np.sum(angularParts[1:,24,24:] * angularParts[:-1,24,24:], axis = 1))
        # exit()
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
    
    def ZernikePolynomialRadialSimple(self, n:int, m:int, rValuesInput = None) -> np.ndarray:
        assert(n>=m>=0)
        assert((n-m)%2 == 0)
        radialPart = np.zeros_like(rValuesInput)
        for k in range(int((n-m)/2)+1):
            sumTerm = (-1)**k
            sumTerm *= factorial(n-k)
            sumTerm *= (rValuesInput/self.maxR)**(n-2*k)
            sumTerm /= factorial(k)
            sumTerm /= factorial((n+m)/2-k)
            sumTerm /= factorial((n-m)/2-k)
            radialPart += sumTerm
        radialPart[rValuesInput > self.maxR] = 0
        normalizationFactor = np.sqrt(2*(n+1)) if m else np.sqrt(n+1)
        return radialPart * normalizationFactor
    
    def OSAANSIIndexToMNIndex(self, OSAANSIIndex:int):
        n = int((np.sqrt(8 * OSAANSIIndex + 1) - 1) / 2)
        m = 2 * OSAANSIIndex - n * (n + 2)
        return m,n
    