
import glob
from tqdm import tqdm
import pandas as pd
import sidetable as stb


for csvFile in tqdm(glob.glob("measurements_*/*.csv")):
    if glob.glob(csvFile.split(".")[0] + "noNan.csv") or "noNan" in csvFile:
        continue
    print(csvFile)
    df = pd.read_csv(csvFile)
    dfNoNan = pd.read_csv(csvFile).dropna()
    dfNoNan.to_csv(csvFile.split(".")[0] + "noNan.csv", index=False)

    