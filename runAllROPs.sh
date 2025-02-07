#!/bin/bash

# Automatically find all folders starting with 'PredROP' and submit the script.sh file in each of them
for folder in PredROP*/; do
    echo "Processing $folder..."
    if [ -f "${folder}ROP.sh" ]; then
        (cd "$folder" && sbatch ROP.sh)
        echo "Submitted ROP.sh in $folder"
    else
        echo "No ROP.sh found in $folder, skipping."
    fi
done

echo "Processing TotalGridROP/..."
if [ -f "TotalGridROP/ROP.sh" ]; then
    (cd "TotalGridROP/" && sbatch ROP.sh)
    echo "Submitted ROP.sh in TotalGridROP/"
else
    echo "No ROP.sh found in TotalGridROP/, skipping."
fi

echo "Processing TotalPosROP/..."
if [ -f "TotalPosROP/ROP.sh" ]; then
    (cd "TotalPosROP/" && sbatch ROP.sh)
    echo "Submitted ROP.sh in TotalPosROP/"
else
    echo "No ROP.sh found in TotalPosROP/, skipping."
fi