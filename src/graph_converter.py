import torch
from torch_geometric.data import Data
from rdkit import Chem

# We define a list of common atoms we expect to see in drugs.
# One-hot encoding prevents the AI from thinking Oxygen (8) is "greater" than Carbon (6).
PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']


def one_hot_encoding(value, choices):
    """Creates a binary list [0, 1, 0, 0...] where 1 indicates the match."""
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1  # Mark as 'Unknown'
    return encoding


def smiles_to_graph(smiles):
    """Converts a SMILES string into a PyTorch Geometric Graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # In case a SMILES string is invalid

    # 1. NODE FEATURES (x)
    node_features = []
    for atom in mol.GetAtoms():
        # Get atom symbol and one-hot encode it
        atom_symbol = atom.GetSymbol()
        atom_features = one_hot_encoding(atom_symbol, PERMITTED_ATOMS)

        # Add a couple more physical properties
        atom_features.append(atom.GetDegree())  # Number of bonds
        atom_features.append(atom.GetFormalCharge())  # Electrical charge
        atom_features.append(int(atom.GetIsAromatic()))  # Is it in an aromatic ring?

        node_features.append(atom_features)

    # Convert our list of lists into a PyTorch Tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. EDGE INDEX (Connectivity) & EDGE FEATURES
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        # Get the IDs of the two atoms connected by this bond
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Bond Type (Single, Double, Triple, Aromatic)
        bond_type = bond.GetBondTypeAsDouble()

        # Molecules are non-directional (if A is bonded to B, B is bonded to A).
        # So we must add the connection in BOTH directions for the GNN.
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [[bond_type], [bond_type]]

    # PyG expects edge_index to be shape [2, num_edges]. We transpose (.t()) to fix this.
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handles edge-case molecules with no bonds (single ions)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # 3. CREATE THE PYG DATA OBJECT
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data


if __name__ == "__main__":
    # Let's test it on Aspirin!
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    graph = smiles_to_graph(aspirin_smiles)

    print("--- PYTORCH GEOMETRIC GRAPH ---")
    print(f"Molecule: Aspirin")
    print(f"Graph Object: {graph}")
    print(f"Node matrix (x) shape: {graph.x.shape} -> (Num Atoms, Num Features)")
    print(f"Edge index shape: {graph.edge_index.shape} -> (2, Num Directed Edges)")
    print(f"Edge attribute shape: {graph.edge_attr.shape} -> (Num Directed Edges, Num Features)")