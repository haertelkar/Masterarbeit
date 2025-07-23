import pandas as pd
import sys
import tqdm         

# if an argument is appended create the variable Folder_Appendix 
if len(sys.argv) > 1:
    Folder_Appendix = sys.argv[1]
else:
    Folder_Appendix = ""

tqdm.tqdm.pandas(desc="Processing files")


for test_or_train in ["train", "test"]:
    labels = pd.read_csv(f"measurements_{test_or_train}{Folder_Appendix}/labels.csv",  sep=",")
    labels.drop(columns = [f"element{i}" for i in range(10)], inplace = True, errors="ignore")
    labels.to_csv(f"measurements_{test_or_train}{Folder_Appendix}/labels_only_Dist.csv", index = False)
    labels = pd.read_csv(f"measurements_{test_or_train}{Folder_Appendix}/labels.csv",  sep=",")
    labels.drop(columns = [f"xAtomRel{a}" for a in range(10)]+[f"yAtomRel{b}" for b in range(10)], inplace = True, errors="ignore")
    labels.to_csv(f"measurements_{test_or_train}{Folder_Appendix}/labels_only_Elem.csv", index = False)

    