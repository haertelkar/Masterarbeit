import shutil
import os
from tqdm import tqdm
import sys

def cleanUp(directory = "", printEmptiedDirs = False, cleanCurrentDir = False):
    emptyDirs = set()
    measurementDirs = ["measurements_test","measurements_train"]
    zmD = [os.path.join("Zernike",d) for d in measurementDirs]
    fmD = [os.path.join("FullPixelGridML",d) for d in measurementDirs]
    measurementDirs+= zmD + fmD
    if cleanCurrentDir: measurementDirs = [directory]
    currentMainDir = os.path.join(os.getcwd(),directory)
    for filename in tqdm(os.listdir(currentMainDir), desc = f"deleting content {currentMainDir}", leave = False):
        if ".progress" in filename or ("progress" in filename and ".txt" in filename):
            os.remove(os.path.join(currentMainDir, filename))
            emptyDirs.add(currentMainDir)
        if ".out" == filename[-4:]:
            #create folder for old slurm files if it does not exist
            if not os.path.exists(os.path.join(currentMainDir, "oldSlurmFiles")):
                os.mkdir(os.path.join(currentMainDir, "oldSlurmFiles"))
            shutil.move(os.path.join(currentMainDir, filename), os.path.join(currentMainDir, "oldSlurmFiles", filename))
            emptyDirs.add(currentMainDir)
    for direct in measurementDirs:
        dirFull = os.path.join(currentMainDir, direct)
        if not os.path.exists(dirFull):
            continue

        for filename in tqdm(os.listdir(dirFull), desc = f"deleting content {dirFull}", leave = False):
            if ".png" in filename or ".npy" in filename or ".csv" in filename or ".hdf5" in filename or ".h5" in filename or ".progress" in filename or ("progress" in filename and ".txt" in filename):
                os.remove(os.path.join(dirFull, filename))
                emptyDirs.add(dirFull)
    


    # for dirpath, dirnames, filenames in tqdm(os.walk(), desc = "Going through folders",disable= not printEmptiedDirs):
    #     print(dirpath[dirpath.find("Masterarbeit"):].count(os.sep))
    #     print(dirpath[dirpath.find("Masterarbeit"):])
    #     if dirpath[dirpath.find("Masterarbeit"):].count(os.sep)>2:
    #         continue 
    #     if "measurements" not in dirpath:
    #         continue
        
    #     for filename in tqdm(filenames, desc = f"deleting {dirpath}", leave = False):
    #         if ".npy" in filename or ".csv" in filename:
    #             os.remove(os.path.join(dirpath, filename))
    #             emptyDirs.add(dirpath)

    if printEmptiedDirs:
        print("Following directories were emptied:")
        for dir in emptyDirs:
            print(dir)

if __name__ == "__main__":
    if len(sys.argv) ==2:
        cleanUp(directory=sys.argv[1], printEmptiedDirs=True, cleanCurrentDir=True)
    else:
        cleanUp(printEmptiedDirs=True)