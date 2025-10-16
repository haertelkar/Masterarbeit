import os
import sys
from tqdm import tqdm
import shutil

import glob
if len(sys.argv) == 1: raise ValueError("Please specify a folder path as an argument.")
folder = sys.argv[1]  
os.makedirs(folder, exist_ok=True)
os.makedirs(os.path.join(folder, "images"), exist_ok=True)
os.makedirs(os.path.join(folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(folder, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(folder, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(folder, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(folder, "labels", "val"), exist_ok=True)

for cnt, groundTruthFile in enumerate(tqdm(glob.glob("/data/scratch/haertelk/Masterarbeit/ROPResults/YOLORUN_2025_07_06_8s_-50def/**/groundTruth.txt", recursive=True), desc="Processing ground truth files")):
    label = groundTruthFile
    path = os.path.dirname(groundTruthFile)
    image = os.path.join(path, "Predictions_No_Border.png")
    if not os.path.exists(image):
        print(f"Image file {image} does not exist. Skipping...")
        continue
    train_or_val = "train" if cnt % 10 != 0 else "val"
    shutil.copy(image, os.path.join(folder, "images", train_or_val, f"{cnt}.png"))
    shutil.copy(label, os.path.join(folder, "labels", train_or_val, f"{cnt}.txt"))
