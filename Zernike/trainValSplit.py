from glob import glob
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Directory containing the CSV files


# Iterate over all CSV files in the directory
for filename in tqdm(glob("measurements_train/labels*.csv"), desc="Splitting train and validation sets"):
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Calculate the number of rows for train and validation sets
    train_rows = int(len(df) * 0.75)
    vali_rows = len(df) - train_rows
    
    # Split the dataframe into train and validation sets
    train_df = df[:train_rows]
    vali_df = df[train_rows:]
    tqdm.write(f"{filename}:\nTrain set: {len(train_df)} rows, Validation set: {len(vali_df)} rows")

    # Save the train and validation sets to new CSV files
    train_df.to_csv("/".join(filename.split("/")[:-1]) + "/train_" + filename.split("/")[-1], index=False)
    tqdm.write(f"saved train date")
    vali_df.to_csv("/".join(filename.split("/")[:-1]) + "/vali_" + filename.split("/")[-1], index=False)
    tqdm.write(f"saved vali date")
    
    # Create histograms for the train datasets
    if "labels_only_Dist_" in filename:
        tqdm.write(f"Creating distance histograms")
        hist, xbins, ybins = np.histogram2d(train_df.iloc[:,1], train_df.iloc[:,2], bins=100)
        hist = hist + 1 #to make zero the lowest value
        xbinPositions = np.digitize(train_df.iloc[:,1], xbins, right = True)-1
        ybinPositions = np.digitize(train_df.iloc[:,2], ybins, right = True)-1
        weights = list(1/(hist[xbinPositions[cnt], ybinPositions[cnt]]) for cnt in range(len(train_df)))
        pd.DataFrame(weights).to_csv("/".join(filename.split("/")[:-1]) + "/weights_" + filename.split("/")[-1], index=False, header=False)

    elif "labels_only_Elem_" in filename:
        tqdm.write(f"Creating element histograms")
        hist, xbins = np.histogram(train_df.iloc[:,1], bins=100)
        hist = hist + 1 #to make zero the lowest value
        weightsBinPosititions = np.digitize(train_df.iloc[:,1], xbins, right = True)-1
        weights = list(1/(hist[weightsBinPosititions[cnt]]) for cnt in range(len(train_df)))
        pd.DataFrame(weights).to_csv("/".join(filename.split("/")[:-1]) + "/weights_" + filename.split("/")[-1], index=False, header=False)