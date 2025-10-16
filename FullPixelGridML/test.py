import numpy as np
silence = False  # Set to True to suppress output
choosenCoords = np.arange(500, dtype=int) #dummy array to avoid errors in the next line

splitting = min(5, len(choosenCoords)) #split the calculation into smaller chunks to avoid memory issues   
for cnt in range(len(choosenCoords) // splitting):
    
    chosenCoordsSplit = choosenCoords[cnt*splitting:(cnt+1)*splitting]
    print(f"Chosen coordinates split: {chosenCoordsSplit}")
    if len(chosenCoordsSplit) == 0: continue
    if not silence:
        print(f"Calculating {cnt * splitting} to {(cnt + 1) * splitting} of  {len(choosenCoords)} coordinates")