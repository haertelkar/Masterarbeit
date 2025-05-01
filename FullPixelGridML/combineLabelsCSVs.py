import glob
import os
import pandas as pd
from tqdm import tqdm
import h5py
import sys

def combineLabelsAndCSV(workingDir, FolderAppendix = ""):
    for testOrTrain in ["train", "test"]:
        filesToDelete = []
        labelsToConcat = []
        indicesToConcat = []
        filesToIgnore = []
        df = pd.DataFrame()
        noOfLabelCSVs = 0
        if len(os.listdir(os.path.join(workingDir,f"measurements_{testOrTrain}{FolderAppendix}"))) <= 3:
            continue
        with h5py.File(os.path.join(workingDir,f"measurements_{testOrTrain}{FolderAppendix}",'temp_copy.h5'),mode='w') as h5fw:
            for file in tqdm(os.listdir(os.path.join(workingDir, f"measurements_{testOrTrain}{FolderAppendix}")), desc= f"Searching for labels in measurements_{testOrTrain}{FolderAppendix}"):
                if (".csv" == file[-4:]) and ("labels" in file):
                    labelsToConcat.append(file)
                    if file != "labels.csv": filesToDelete.append(file)
                    noOfLabelCSVs +=1
                # if (".csv" == file[-4:]) and ("Indices" in file):
                #     indicesToConcat.append(file)
                #     if file != "fractionOfNonZeroIndices.csv": filesToDelete.append(file)
                if ".hdf5" == file[-5:]:
                    try:
                        #if "training_data.hdf5" != file: filesToDelete.append(file)
                        with h5py.File(os.path.join(*[workingDir, f"measurements_{testOrTrain}{FolderAppendix}",file]),'r') as h5fr:
                            for datasetName in tqdm(h5fr.keys(), desc = f"Going through file", leave = False):  
                                try:
                                    h5fw[datasetName] = h5py.ExternalLink(file, datasetName)      
                                except RuntimeError as e:
                                    print(f"Warning in {file}\n{e}\n\n IGNORED FOR NOW")
                    except OSError as e:
                        filesToIgnore.append(file)
                    except Exception as e:
                        print(f"Problem in {file}")
                        filesToIgnore.append(file)
                        print(e)
                        print("ignored for now")

        
                
        print(f"Found {noOfLabelCSVs} files. Combining into one file. {len(filesToIgnore)} .hdf5-files were corrupted and are deleted with their csv.-counterparts")        

        cntCor = 0
        # if len(labelsToConcat) != 0:
        #     for file in indicesToConcat:
        #         id = "_".join(".".join(file.split(".")[:-1]).split("_")[1:])
        #         if f"{id}.hdf5" in filesToIgnore:
        #             continue
        #         data = pd.read_csv(os.path.join(f"measurements_{testOrTrain}",file))
        #         df = pd.concat([df, data], axis = 0)
        #     df.to_csv(os.path.join(f"measurements_{testOrTrain}","fractionOfNonZeroIndices.csv"), index= False)

        for file in labelsToConcat:
            id = "_".join(".".join(file.split(".")[:-1]).split("_")[1:])
            if f"{id}.hdf5" in filesToIgnore:
                cntCor += 1
                continue
            data = pd.read_csv(os.path.join(f"measurements_{testOrTrain}{FolderAppendix}",file), converters={'File Name': str})
            df = pd.concat([df, data], axis = 0)
        df.to_csv(os.path.join(f"measurements_{testOrTrain}{FolderAppendix}","labels.csv"), index= False)

        print(f"Of all corrupted files, {cntCor} had labels files that needed to be deleted.")
        for fileToDelete in filesToDelete:
            os.remove(os.path.join(f"measurements_{testOrTrain}{FolderAppendix}", fileToDelete))

        os.rename(os.path.join(f"measurements_{testOrTrain}{FolderAppendix}","temp_copy.h5"), os.path.join(f"measurements_{testOrTrain}{FolderAppendix}","training_data.hdf5"))

if __name__ == "__main__":
    FolderAppendix = ""
    if len(sys.argv) > 1:
        FolderAppendix = sys.argv[1]
    else:
        FolderAppendix = ""
    combineLabelsAndCSV(".", FolderAppendix)