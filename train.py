import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import SimpleNN

def masked_mse(y_pred, y_true, threshold=1e-5):
    mask = y_true > threshold
    return ((y_pred[mask] - y_true[mask]) ** 2).mean().item() if mask.any() else 0.0

def masked_mae(y_pred, y_true, threshold=1e-5):
    mask = y_true > threshold
    return (torch.abs(y_pred[mask] - y_true[mask])).mean().item() if mask.any() else 0.0


def masked_mse_loss(y_pred, y_true, threshold=0):
    mask = y_true > threshold
    return ((y_pred[mask] - y_true[mask]) ** 2).mean()



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

X = np.load('data/input.npy')
y = np.load('data/output.npy')

# 2. Log transform for better scaling
X = np.log1p(X)
y = np.log1p(y)

# 3. Normalize (optional but helpful)
# x_scaler = StandardScaler()
# y_scaler = StandardScaler()

# X = x_scaler.fit_transform(X)
# y = y_scaler.fit_transform(y)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 6. Create DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 7. Define the model

model = SimpleNN()

# 8. Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 9. Training loop
for epoch in range(1000):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f}")

# 10. Evaluate on test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

    # Compute and print metrics
    val_mse = masked_mse(y_pred, y_test)

    print(f"Epoch {epoch+1:03d} | Train Loss: {total_loss:.4f} | Masked MSE: {val_mse:.6f} | Masked MAE: {val_mse:.6f}")


# Save just the weights (recommended)
torch.save(model.state_dict(), "compound_to_oil_model.pth")
