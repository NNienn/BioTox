import pandas as pd
import os
import torch
from torch_geometric.data import Dataset
# Import the translator we built in the last step!
from graph_converter import smiles_to_graph
# We will use PyTorch Geometric's built-in MoleculeNet downloader
from torch_geometric.datasets import MoleculeNet


def download_and_prepare_data():
    print("Downloading Tox21 Dataset...")

    # This automatically downloads the Tox21 dataset into your data/ folder
    data_path = os.path.join("..", "data")
    dataset = MoleculeNet(root=data_path, name='Tox21')

    print(f"\nDataset Downloaded!")
    print(f"Total molecules in dataset: {len(dataset)}")
    print(f"Number of target labels per molecule: {dataset.num_classes}")

    # Let's look at the very first molecule in the dataset
    first_molecule = dataset[0]
    print("\n--- FIRST MOLECULE IN TOX21 ---")
    print(first_molecule)

    # Tox21 measures toxicity across 12 different biological receptors.
    # We just want to see the labels (0 for safe, 1 for toxic, NaN for untested)
    print(f"Toxicity Labels for molecule 0: {first_molecule.y}")


if __name__ == "__main__":
    download_and_prepare_data()