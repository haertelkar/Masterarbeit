fileName = "results_ZernikeNormal_evaluation_ZernikeNormal_0607_OnlyDist_epoch=30-step=50840_randomData.csv"

import pandas as pd
#import the file as pd.dataframe and add a new column at the beginning and in the middle
df = pd.read_csv("testDataEval/"+fileName)  # read the file
df.insert(0, 'element', 0)  # add a new column at the beginning
df.insert(3, 'element1', 0)  # add a new column in the middle
df.to_csv(f"testDataEval/AppendedColumn_{fileName}", index=False)  # save the file