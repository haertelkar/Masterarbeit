import shutil
import os
from tqdm import tqdm

emptyDirs = set()

for dirpath, dirnames, filenames in tqdm(os.walk(os.getcwd())):
    if "measurements" not in dirpath:
        continue
    for filename in filenames:
        if ".npy" in filename or ".csv" in filename:
            os.remove(os.path.join(dirpath, filename))
            emptyDirs.add(dirpath)


print("Following directories were emptied:")

for dir in emptyDirs:
    print(dir)