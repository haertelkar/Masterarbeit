from FullPixelGridML.SimulateTilesOneFile import generateDiffractionArray, createAllXYCoordinates, saveAllDifPatterns

DIMTILES = 10
BFDdiameter = 18

saveAllDifPatterns(DIMTILES, DIMTILES, None, 1, None, BFDdiameter, processID = 99999, silence = False, maxPooling = 1, structure = "grapheneC", fileWrite = False)