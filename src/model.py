import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT_Toxicity_Predictor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GAT_Toxicity_Predictor, self).__init__()

        # 1. The Graph Attention Layers
        # This is where the atoms talk to each other and pay "attention" to important bonds
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)

        # 2. The Linear Classifier
        # This takes the final graph summary and squashes it into 12 toxicity predictions
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # --- ATOM LEVEL LEARNING ---
        # Layer 1: Atoms look at immediate neighbors
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # ReLU adds non-linearity (helps the AI learn complex patterns)

        # Layer 2: Atoms look at neighbors of neighbors
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Layer 3: Atoms look at neighbors of neighbors of neighbors
        x = self.conv3(x, edge_index)

        # --- GRAPH LEVEL LEARNING ---
        # Right now, 'x' is a list of atoms. We need to evaluate the WHOLE molecule.
        # global_mean_pool takes the average of all atom states to represent the whole drug
        x = global_mean_pool(x, batch)

        # --- FINAL PREDICTION ---
        # Pass the whole-drug representation through the final layer
        # We don't use an activation function here because we will use BCEWithLogitsLoss later
        out = self.lin(x)

        return out


if __name__ == "__main__":
    # Let's test if the AI can process a dummy molecule!
    print("Initializing untrained GAT Model...")

    # Based on our data output: 9 features per atom, predicting 12 labels
    model = GAT_Toxicity_Predictor(num_node_features=9, hidden_channels=64, num_classes=12)

    print(model)

    # Count the parameters (how many "neurons" the AI has to learn with)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params}")
    print("If this number is low (under 100k), it means it will train easily on your CPU/4GB GPU!")