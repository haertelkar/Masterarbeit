resultVectorLength = 0 
numberOfN = 40
for n in range(1,numberOfN + 1):
    for mShifted in range(2*n+1):
        m = mShifted - n
        if (n-m)%2 != 0:
            continue
        resultVectorLength += 1

print("number of Zernike Moments",resultVectorLength)