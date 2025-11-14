import argparse
import os
import pandas as pd
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt

def plot_distances(data_dir):
    """
    Reads atom coordinates from a CSV file and plots histograms of the
    pairwise distances between atoms in x and y dimensions.

    Args:
        data_dir (str): The path to the directory containing 'labels.csv'.
    """
    file_path = os.path.join(data_dir, 'labels.csv')

    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    num_atoms = 9  # Based on the column names from xAtomRel0 to xAtomRel8

    # Extract x and y coordinates
    x_cols = [f'xAtomRel{i}' for i in range(num_atoms)]
    y_cols = [f'yAtomRel{i}' for i in range(num_atoms)]

    x_coords = df[x_cols].values
    y_coords = df[y_cols].values

    x_distances = []
    y_distances = []

    # Generate all unique pairs of indices (0,1), (0,2), ..., (7,8)
    indices = list(combinations(range(num_atoms), 2))

    # Calculate pairwise distances for each row
    for i, j in indices:
        # Calculate absolute difference for all rows at once
        x_dist = np.abs(x_coords[:, i] - x_coords[:, j])
        y_dist = np.abs(y_coords[:, i] - y_coords[:, j])
        
        # Extend the lists with the new distances
        x_distances.extend(x_dist)
        y_distances.extend(y_dist)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    # Histogram for x-distances
    axs[0].hist(x_distances, bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title('Distribution of Pairwise X-Distances')
    axs[0].set_xlabel('Absolute Distance (xAtomRel)')
    axs[0].set_ylabel('Frequency')
    axs[0].grid(axis='y', alpha=0.75)

    # Histogram for y-distances
    axs[1].hist(y_distances, bins=50, color='salmon', edgecolor='black')
    axs[1].set_title('Distribution of Pairwise Y-Distances')
    axs[1].set_xlabel('Absolute Distance (yAtomRel)')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(axis='y', alpha=0.75)

    plt.suptitle('Histograms of Relative Atom Distances')
    plt.savefig("atom_distances_histogram.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot histograms of relative atom distances from a labels.csv file."
    )
    parser.add_argument(
        'subfolder',
        type=str,
        help="The subfolder containing the 'labels.csv' file."
    )
    args = parser.parse_args()

    plot_distances(args.subfolder)