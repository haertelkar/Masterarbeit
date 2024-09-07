import pandas as pd

for test_or_train in ["train", "test"]:
    labels = pd.read_csv(f"measurements_{test_or_train}/labels.csv",  sep=",")
    labels.drop(columns = ["element1", "element2", "element3", "element4"], inplace = True, errors="ignore")
    labels.to_csv(f"measurements_{test_or_train}/labels_only_Dist.csv", index = False)
    labels = pd.read_csv(f"measurements_{test_or_train}/labels.csv",  sep=",")
    labels.drop(columns = ["xAtomRel1","xAtomRel2","xAtomRel3","xAtomRel4","yAtomRel1","yAtomRel2","yAtomRel3","yAtomRel4"], inplace = True, errors="ignore")
    labels.to_csv(f"measurements_{test_or_train}/labels_only_Elem.csv", index = False)