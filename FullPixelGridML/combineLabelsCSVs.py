import glob
import os
import pandas as pd
from tqdm import tqdm
import h5py



for testOrTrain in ["train", "test"]:
    filesToDelete = []
    labelsToConcat = []
    filesToIgnore = []
    df = pd.DataFrame()
    noOfCSVs = 0
    with h5py.File(os.path.join(f"measurements_{testOrTrain}",'temp_copy.h5'),mode='w') as h5fw:
        for file in tqdm(os.listdir(f"measurements_{testOrTrain}"), desc= f"Searching for labels in measurements_{testOrTrain}"):
            if (".csv" == file[-4:]) and ("labels" in file):
                labelsToConcat.append(file)
                if file != "labels.csv": filesToDelete.append(file)
                noOfCSVs +=1
            if ".hdf5" == file[-5:]:
                try:
                    if "training_data.hdf5" != file: filesToDelete.append(file)
                    with h5py.File(os.path.join(f"measurements_{testOrTrain}",file),'r') as h5fr:
                        for obj in h5fr.keys():        
                            h5fr.copy(obj, h5fw)
                except OSError as e:
                    filesToIgnore.append(file)
            
            
    print(f"Found {noOfCSVs} files. Combining into one file. {len(filesToIgnore)} .hdf5-files were corrupted and are deleted with their csv.-counterparts")
    cntCor = 0
    for file in labelsToConcat:
        id = "_".join(".".join(file.split(".")[:-1]).split("_")[1:])
        if id + ".hdf5" in filesToIgnore:
            cntCor += 1
            continue
        data = pd.read_csv(os.path.join(f"measurements_{testOrTrain}",file))
        df = pd.concat([df, data], axis = 0)
    df.to_csv(os.path.join(f"measurements_{testOrTrain}","labels.csv"), index= False)

    print("Successfully deleted {cntCor} deleted files.")
    for fileToDelete in filesToDelete:
        os.remove(os.path.join(f"measurements_{testOrTrain}", fileToDelete))

    os.rename(os.path.join(f"measurements_{testOrTrain}","temp_copy.h5"), os.path.join(f"measurements_{testOrTrain}","training_data.hdf5"))