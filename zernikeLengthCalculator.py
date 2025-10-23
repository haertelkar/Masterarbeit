def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


for numberOfN in range(6,17):
    resultVectorLength = 0 
    for n in range(1,numberOfN + 1):
        for mShifted in range(2*n+1):
            m = mShifted - n
            if (n-m)%2 != 0:
                continue
            resultVectorLength += 1
    print("-"*10)
    print(numberOfN)
    print("number of Zernike Moments",resultVectorLength)
    print("prime factor of result + 2", prime_factors(resultVectorLength+2))
    print("-"*10)