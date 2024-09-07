import numpy as np
import pandas as pd
from tqdm import tqdm

with tqdm(desc = "updating Files", total= 16) as pbar:
    for test_or_train in ["train", "test"]:
        labels = pd.read_csv(f"measurements_{test_or_train}/labels.csv",  sep=",")
        labels.drop(columns = ["element1", "element2", "element3", "element4"], inplace = True, errors="ignore")
        for i in range(1,5):
            allAtomNos = [1,2,3,4]
            allAtomNos.remove(i)
            labels.drop(columns = [f"{xy}AtomRel{atomNo}" for atomNo in allAtomNos for xy in ["x","y"]], errors="ignore").to_csv(f"measurements_{test_or_train}/labels_only_Dist_{i}.csv", index = False)
            pbar.update(1)

    for test_or_train in ["train", "test"]:
        labels = pd.read_csv(f"measurements_{test_or_train}/labels.csv",  sep=",")
        labels.drop(columns = [f"{xy}AtomRel{cnt}" for cnt in [1,2,3,4] for xy in ["x","y"]], inplace = True, errors="ignore")
        for i in range(1,5):
            allAtomNos = [1,2,3,4]
            allAtomNos.remove(i)
            labels.drop(columns = [f"element{atomNo}" for atomNo in allAtomNos], errors="ignore").to_csv(f"measurements_{test_or_train}/labels_only_Elem_{i}.csv", index = False)
            pbar.update(1)