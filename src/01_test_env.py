import torch
import torch_geometric
import rdkit
from rdkit import Chem

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"RDKit version: {rdkit.__version__}")
print(f"Is CUDA (GPU) available? {torch.cuda.is_available()}")

# 1. Define a molecule using a SMILES string (This is Benzene, a ring of 6 Carbons)
smiles_string = "C1=CC=CC=C1"

# 2. Convert text to an RDKit Molecule Object
molecule = Chem.MolFromSmiles(smiles_string)

# 3. Extract the basic graph information
num_atoms = molecule.GetNumAtoms()
num_bonds = molecule.GetNumBonds()

print("\n--- MOLECULE TEST ---")
print(f"Molecule: Benzene ({smiles_string})")
print(f"Number of Nodes (Atoms): {num_atoms}")
print(f"Number of Edges (Bonds): {num_bonds}")

# 4. Look at the specific atoms
print("\nAtom Breakdown:")
for atom in molecule.GetAtoms():
    print(f"Atom Index {atom.GetIdx()}: Type {atom.GetSymbol()}, Atomic Number {atom.GetAtomicNum()}")