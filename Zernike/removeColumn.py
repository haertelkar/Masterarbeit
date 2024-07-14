import pandas as pd

for test_or_train in ["test", "train"]:
    labels = pd.read_csv(f"measurements_{test_or_train}/labels.csv",  sep=",")
    labels.drop(columns = ["xAtomRel","yAtomRel"], inplace = True, errors="ignore")
    labels.to_csv(f"measurements_{test_or_train}/labels_OnlyElem.csv", index = False)