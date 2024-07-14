import pandas as pd

for test_or_train in ["test", "train"]:
    print(f"Processing {test_or_train} data")
    labels = pd.read_csv(f"measurements_{test_or_train}/labels.csv",  sep=",")
    labels["xAtomRel"] = (labels["xAtomRel"]/(labels["xAtomRel"]**3 + 1e-10)).apply(lambda x: max(0.5,min(x, 100)))
    labels["yAtomRel"] = (labels["xAtomRel"]/(labels["yAtomRel"]**3 + 1e-10)).apply(lambda x: max(0.5,min(x, 100)))
    labels.rename(columns={"xAtomRel": "xAtomRelOverDistCub", "yAtomRel": "yAtomRelOverDistCub"}, inplace=True)
    labels.to_csv(f"measurements_{test_or_train}/labels_DistOverDistCub.csv", index = False)