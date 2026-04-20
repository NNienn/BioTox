import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from model import GAT_Toxicity_Predictor  # Importing your AI!


def main():
    # 1. HARDWARE SETUP
    # This automatically detects if your GPU is fixed. If not, it safely uses your CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}\n")

    # 2. LOAD AND SPLIT DATA
    print("Loading Dataset...")
    data_path = os.path.join("..", "data")
    dataset = MoleculeNet(root=data_path, name='Tox21')

    # Shuffle the data randomly
    dataset = dataset.shuffle()

    # Split: 80% for training (learning), 20% for testing (final exam)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    # DataLoader batches molecules together so the CPU processes 32 at a time (highly efficient!)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training on {len(train_dataset)} molecules.")
    print(f"Testing on {len(test_dataset)} molecules.\n")

    # 3. INITIALIZE AI, LOSS, AND OPTIMIZER
    model = GAT_Toxicity_Predictor(
        num_node_features=dataset.num_node_features,
        hidden_channels=64,
        num_classes=dataset.num_classes
    ).to(device)

    # Adam is the standard optimizer that adjusts the AI's parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # FIX #1: COMPUTE pos_weight TO HANDLE CLASS IMBALANCE
    # Tox21 is ~95% "safe" labels. Without this, the model learns to always
    # predict "safe" and still scores 95% accuracy — a complete lie.
    # pos_weight tells the loss: "penalise missed toxic predictions much harder."
    print("Computing class weights to fix imbalance...")
    all_labels = torch.cat([data.y for data in train_dataset], dim=0)
    valid_mask = ~torch.isnan(all_labels)
    valid_labels = all_labels[valid_mask]
    neg_count = (valid_labels == 0).sum().float()
    pos_count = (valid_labels == 1).sum().float()
    pos_weight = (neg_count / pos_count).to(device)
    print(f"  Negative (safe) labels : {int(neg_count)}")
    print(f"  Positive (toxic) labels: {int(pos_count)}")
    print(f"  pos_weight applied     : {pos_weight:.2f}x\n")

    # BCEWithLogitsLoss is perfect for multi-label yes/no predictions.
    # reduction='none' is CRUCIAL so we can mask out the 'nan' values manually.
    # pos_weight corrects the imbalance so the model actually learns toxicity.
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    # FIX #2: ADD A LEARNING RATE SCHEDULER
    # After 20 epochs the loss often plateaus. Halving the LR every 20 epochs
    # lets the model make finer adjustments and keep improving.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 4. THE TRAINING LOOP FUNCTION
    def train_one_epoch():
        model.train()  # Tell the AI it's in learning mode
        total_loss = 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)

            # Step A: Clear old memory
            optimizer.zero_grad()

            # Step B: AI makes its guess
            # We add .float() because the AI needs decimals, not raw integers!
            predictions = model(batch_data.x.float(), batch_data.edge_index, batch_data.batch)

            # Step C: The "Masking" Trick for missing lab data (NaNs)
            true_labels = batch_data.y
            is_valid = ~torch.isnan(true_labels)  # Find where labels actually exist

            # Step D: Calculate how wrong the AI is (Loss), ONLY on valid data
            loss_matrix = criterion(predictions[is_valid], true_labels[is_valid])
            loss = loss_matrix.mean()  # Get the average loss for the batch

            # Step E: Backpropagation (Learn from mistakes)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    # 5. RUN THE TRAINING OVER MULTIPLE EPOCHS
    # FIX #3: INCREASED EPOCHS FROM 15 → 50
    # GATs on molecular data need more passes to converge properly.
    print("Starting Training...")
    epochs = 50

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch()
        scheduler.step()  # Decay LR every 20 epochs as configured above
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{epochs} | Training Loss: {loss:.4f} | LR: {current_lr:.6f}")

    print("\nTraining Complete! The AI has learned the chemistry!")

    # 6. THE FINAL EXAM (Test Loop)
    # FIX #4: REPLACE ACCURACY WITH ROC-AUC
    # Accuracy is MEANINGLESS on imbalanced data — a model predicting all-zeros
    # scores ~95% accuracy while detecting zero toxic compounds.
    # ROC-AUC measures real discrimination ability:
    #   0.5 = random guess, 0.75+ = decent, 0.85+ = good
    print("\nEvaluating on Test Data...")
    model.eval()  # Tell the AI it is taking a test (turns off learning)

    test_loss = 0
    all_preds = []
    all_true_labels = []

    # torch.no_grad() disables calculus/gradients. It saves massive amounts of memory!
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)

            # AI makes its predictions (raw logits)
            predictions = model(batch_data.x.float(), batch_data.edge_index, batch_data.batch)

            # Mask out the NaNs for loss calculation
            true_labels = batch_data.y
            is_valid = ~torch.isnan(true_labels)

            # Calculate Test Loss
            loss_matrix = criterion(predictions[is_valid], true_labels[is_valid])
            test_loss += loss_matrix.mean().item()

            # Collect sigmoid probabilities and true labels for ROC-AUC
            probs = torch.sigmoid(predictions)
            all_preds.append(probs.cpu().numpy())
            all_true_labels.append(true_labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    # Compute per-task ROC-AUC, skipping tasks where both classes don't appear
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_true_labels, axis=0)

    receptors = [
        "Androgen Receptor", "Androgen LBD",
        "Aryl Hydrocarbon Receptor", "Aromatase",
        "Estrogen Receptor", "Estrogen LBD",
        "PPAR-gamma", "Antioxidant Response",
        "ATAD5 (DNA Damage)", "Heat Shock Response",
        "Mitochondrial Stress", "p53 (Cancer/DNA)"
    ]

    print("\nPer-Task ROC-AUC Scores:")
    auc_scores = []
    for i in range(all_labels_np.shape[1]):
        col_labels = all_labels_np[:, i]
        col_preds = all_preds_np[:, i]
        valid = ~np.isnan(col_labels)
        if len(np.unique(col_labels[valid])) == 2:  # Both classes must exist
            auc = roc_auc_score(col_labels[valid], col_preds[valid])
            auc_scores.append(auc)
            flag = "✅" if auc >= 0.75 else "⚠️ "
            print(f"  {flag} {receptors[i]:<30} AUC: {auc:.4f}")
        else:
            print(f"  ⏭️  {receptors[i]:<30} Skipped (only one class in test set)")

    if auc_scores:
        mean_auc = np.mean(auc_scores)
        print(f"\nMean ROC-AUC across all tasks: {mean_auc:.4f}")
        if mean_auc >= 0.80:
            print("🟢 Model quality: GOOD — ready for inference")
        elif mean_auc >= 0.70:
            print("🟡 Model quality: FAIR — may miss some toxic compounds")
        else:
            print("🔴 Model quality: POOR — retrain with more epochs or tune further")

    # 7. MLOPS: SAVE THE AI'S BRAIN
    os.makedirs(os.path.join("..", "models"), exist_ok=True)
    save_path = os.path.join("..", "models", "gat_toxicity_model.pth")

    # state_dict() extracts the tuned weights and saves them to a file
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved successfully to {save_path}!")


if __name__ == "__main__":
    main()