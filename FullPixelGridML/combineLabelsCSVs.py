import os
import pandas as pd
from tqdm import tqdm



for testOrTrain in ["train", "test"]:
    CSVsToDelete = []
    df = pd.DataFrame()
    noOfCSVs = 0
    for file in tqdm(os.listdir(f"measurements_{testOrTrain}"), desc= f"Searching for labels in measurements_{testOrTrain}"):
        if (".csv" in file) and ("labels" in file):
            data = pd.read_csv(os.path.join(f"measurements_{testOrTrain}",file))
            df = pd.concat([df, data], axis = 0)
            if file != "labels.csv": CSVsToDelete.append(file)
            noOfCSVs +=1
    print(f"Found {noOfCSVs} files. Combining into one file.")
    df.to_csv(os.path.join(f"measurements_{testOrTrain}","labels.csv"), index= False)

    for csvToDelete in CSVsToDelete:
        os.remove(os.path.join(f"measurements_{testOrTrain}",csvToDelete))