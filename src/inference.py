import torch
from torch_geometric.utils import from_smiles
from model import GAT_Toxicity_Predictor
import os


def main():
    # 1. HARDWARE & ARCHITECTURE SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We must build the exact same "empty" brain we used in training
    model = GAT_Toxicity_Predictor(num_node_features=9, hidden_channels=64, num_classes=12)

    # 2. LOAD THE LEARNED KNOWLEDGE
    model_path = os.path.join("..", "models", "gat_toxicity_model.pth")
    if not os.path.exists(model_path):
        print("Error: Could not find the saved model. Did you run train.py?")
        return

    # Inject the saved weights into the empty brain
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # CRUCIAL: Tell the AI it is taking a test, not learning.
    # This locks the weights so they don't change.
    model.eval()

    def predict_chemical(smiles, name):
        # 3. TRANSLATE TEXT TO GRAPH
        graph = from_smiles(smiles)
        if graph is None:
            print(f"Invalid SMILES: {name}")
            return

        x = graph.x.float().to(device)
        edge_index = graph.edge_index.to(device)

        # Because we are predicting 1 molecule (instead of a batch of 32),
        # we create a dummy batch vector of all zeros
        batch = torch.zeros(x.shape[0], dtype=torch.long).to(device)

        # 4. THE PREDICTION
        with torch.no_grad():  # No calculus needed for predicting
            raw_output = model(x, edge_index, batch)

            # The AI outputs raw numbers (logits).
            # We use a Sigmoid function to squash them into percentages (0 to 1)
            probabilities = torch.sigmoid(raw_output)[0]

        # 5. PRINT THE REPORT
        # These are the actual 12 biological receptors Tox21 tests for
        receptors = [
            "Androgen (Hormone) Receptor", "Androgen LBD",
            "Aryl Hydrocarbon (Toxin) Receptor", "Aromatase (Enzyme)",
            "Estrogen Receptor", "Estrogen LBD",
            "PPAR-gamma (Metabolism)", "Antioxidant Response",
            "ATAD5 (DNA Damage)", "Heat Shock Response",
            "Mitochondrial Stress", "p53 (Cancer/DNA Damage)"
        ]

        print(f"\n🧪 --- TOXICITY REPORT: {name} ---")
        print(f"SMILES: {smiles}")
        print(f"{'Receptor':<35} {'Risk':>6}  {'Level'}")
        print("-" * 60)

        danger_count = 0
        warning_count = 0

        for i, prob in enumerate(probabilities):
            percent = prob.item() * 100

            # FIX #5: LOWERED THRESHOLD FROM 50% → 30%
            # The model was trained with pos_weight, so it is now calibrated to
            # produce meaningful probabilities below 50%. A 30% threshold
            # catches moderate-risk signals that a 50% cutoff would silently ignore.
            if percent > 50.0:
                print(f"  ⚠️  DANGER  | {receptors[i]:<35} {percent:>5.1f}%")
                danger_count += 1
            elif percent > 30.0:
                print(f"  🔶 WARNING | {receptors[i]:<35} {percent:>5.1f}%")
                warning_count += 1

        print("-" * 60)

        if danger_count == 0 and warning_count == 0:
            print("✅  RESULT: Safe. No toxicity signals detected across 12 receptors.")
        else:
            if danger_count > 0:
                print(f"🚨  RESULT: HIGH RISK — {danger_count} receptor(s) flagged as DANGEROUS (>50%)")
            if warning_count > 0:
                print(f"⚠️   RESULT: MODERATE RISK — {warning_count} receptor(s) flagged as WARNING (30–50%)")

    # ---------------------------------------------------------
    # LET'S TEST SOME REAL WORLD DRUGS!
    # ---------------------------------------------------------

    # 1. Ibuprofen (Active ingredient in Advil)
    predict_chemical("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen")

    # 2. Caffeine (The stimulant in Coffee)
    predict_chemical("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine")

    # 3. Parathion (A highly lethal agricultural pesticide / nerve agent)
    predict_chemical("CCOP(=S)(OCC)OC1=CC=C(C=C1)[N+](=O)[O-]", "Parathion (Lethal Nerve Agent)")

    # 4. BPA (The toxic plastic chemical banned in water bottles)
    predict_chemical("CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O", "Bisphenol A (BPA)")


if __name__ == "__main__":
    main()