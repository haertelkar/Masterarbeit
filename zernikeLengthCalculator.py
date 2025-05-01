resultVectorLength = 0 
numberOfOSAANSIMoments = 32	
for n in range(numberOfOSAANSIMoments + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1

print("number of Zernike Moments",resultVectorLength)