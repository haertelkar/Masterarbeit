import os
import pandas as pd

listOfCSVs = []
df = pd.DataFrame()
for file in os.listdir("measurement_test"):
    if file[:-4] == ".csv" and "labels" in file:
        data = pd.read_csv(file)
        df = pd.concat([df, data], axis = 0)
df.to_csv("labels.csv", index= False)
    