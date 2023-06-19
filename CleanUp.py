import shutil
import os
from tqdm import tqdm

def cleanUp(directory = "", printEmptiedDirs = False):
    emptyDirs = set()

    for dirpath, dirnames, filenames in tqdm(os.walk(os.path.join(os.getcwd(),directory)), disable= not printEmptiedDirs):
        if "measurements" not in dirpath:
            continue
        for filename in filenames:
            if ".npy" in filename or ".csv" in filename:
                os.remove(os.path.join(dirpath, filename))
                emptyDirs.add(dirpath)

    if printEmptiedDirs:
        print("Following directories were emptied:")
        for dir in emptyDirs:
            print(dir)

if __name__ == "__main__":
    cleanUp(printEmptiedDirs=True)