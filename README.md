
# BioTox: Graph Attention Network for Molecular Toxicity Prediction

BioTox is a deep learning pipeline designed to predict chemical toxicity risk across 12 biological receptors using the Tox21 benchmark dataset. By converting chemical SMILES strings into molecular graphs, the system utilizes a Graph Attention Network (GAT) to identify toxicological endpoints with high precision.

## 🚀 Project Overview
Traditional toxicity screening is expensive and slow. BioTox automates this by:
- **Encoding** chemical structures as graphs (Atoms = Nodes, Bonds = Edges).
- **Learning** chemical importance via an Attention Mechanism.
- **Predicting** probabilities for 12 human biological receptors (Nuclear and Stress-Response).

## 🏗️ Architecture
The model uses a 3-layer Graph Attention Network (GAT) built with PyTorch Geometric:
1. **Input Layer**: Processes 9-feature atom vectors (element type, degree, charge, aromaticity).
2. **GAT Layers**: 3 layers that expand features (9 → 64 → 64 → 64) and capture up to a 3-hop neighborhood.
3. **Global Mean Pooling**: Averages atom vectors into a single 64-dimensional molecule representation.
4. **Classification Head**: A linear layer mapping to 12 raw logits, followed by a Sigmoid for risk reporting.

## 📊 Data Pipeline & Encoding
- **Dataset**: Tox21 (7,831 molecules).
- **SMILES to Graph**:
    - **Nodes ($x$):** $N 	imes 9$ feature matrix.
    - **Edges (`edge_index`):** Bidirectional directed edges representing chemical bonds.
- **Class Imbalance Fix**: Uses `BCEWithLogitsLoss` with `pos_weight` to account for the ~95% safe / ~5% toxic data split.

## 🩺 Biological Receptors
The system predicts risk for 12 specific assays:
- **Nuclear Receptors (NR)**: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma.
- **Stress-Response (SR)**: SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.

## 📉 Evaluation
The model is evaluated using **ROC-AUC** per task rather than simple accuracy, ensuring real skill in detecting rare toxic compounds.
- **Target**: >0.75 AUC per receptor.
- **Strong Performance**: >0.85 AUC (competitive with published benchmarks).

## 📂 Codebase Structure
- `model.py`: Defines the `GAT_Toxicity_Predictor` class.
- `train.py`: Training loop with imbalance correction and ROC-AUC evaluation.
- `inference.py`: Generates tiered risk reports (**DANGER**, **WARNING**, **Safe**).
- `graph_converter.py`: Translates SMILES strings into PyG Data objects.

## ⚠️ Known Limitations
- **Scope**: Does not currently model ADME (Absorption, Metabolism, Distribution, Excretion).
- **Features**: Bond types (single/double/triple) are extracted but currently unused by the GAT layers.
- **Context**: In vitro results do not always equate to in vivo clinical toxicity.

---
*Generated based on the BioTox Technical Report.*
README.md
Displaying README.md.
