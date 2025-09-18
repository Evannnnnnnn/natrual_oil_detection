import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from model import SimpleBinaryClassifier
import wandb
from focal_loss import FocalLoss

# ----------------------------
# 1. Device setup
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

wandb.init(
    project="compound_to_oil_binary",
    config={
        "epochs": 100,
        "batch_size": 8,
        "lr": 1e-3,
        "loss": "BCEWithLogitsLoss",
        "model": "SimpleNN"
    }
)


INPUT_SIZE = 110
OUTPUT_SIZE = 5

# ----------------------------
# 2. Load data
# ----------------------------
X = np.load('data/input.npy')         # shape: (n_samples, 110)
y = np.load('data/key_output.npy')       # shape: (n_samples, 235)

# ----------------------------
# 3. Convert input and output to binary presence (0/1)
# ----------------------------
X = np.log1p(X)
y_binary = (y > 1e-5).astype(np.float32)

# ----------------------------
# 4. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2)


np.save('data/X_test_input.npy', X_test)
np.save('data/y_test_key_output.npy', y_test)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

pos_counts = y_train.sum(axis=0)
print(pos_counts)
neg_counts = y_train.shape[0] - pos_counts
class_weights = (neg_counts / pos_counts).clip(0, 100)
class_weights = class_weights.detach().clone().to(device)

# ----------------------------
# 5. Define model
# ----------------------------


model = SimpleBinaryClassifier(INPUT_SIZE, OUTPUT_SIZE).to(device)

# ----------------------------
# 6. Training setup
# ----------------------------
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# ----------------------------
# 7. Dataloader
# ----------------------------
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True)

# ----------------------------
# 8. Training loop
# ----------------------------
for epoch in range(wandb.config["epochs"]):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_test)
        val_loss = loss_fn(val_pred, y_test).item()
        val_probs = torch.sigmoid(val_pred)
        threshold = 0.1
        val_bin = (val_probs > threshold).float()

        precision = precision_score(y_test.cpu(), val_bin.cpu(), average='samples', zero_division=0)
        recall = recall_score(y_test.cpu(), val_bin.cpu(), average='samples', zero_division=0)
        f1 = f1_score(y_test.cpu(), val_bin.cpu(), average='samples', zero_division=0)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss / len(train_loader),
        "val_loss": val_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(f"Val_probs: {threshold} Val_loss: {val_loss} Epoch {epoch+1:03d} | Train Loss: {total_loss / len(train_loader):.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")


# ----------------------------
# 9. Save model
# ----------------------------
torch.save(model.state_dict(), "key_cas_key_oil_compound_to_oil_binary_model.pth")
wandb.save("key_cas_key_oils_compound_model.pth")