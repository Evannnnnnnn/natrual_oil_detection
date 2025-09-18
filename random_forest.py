import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Load the data
X = np.load('data/input.npy')  # shape: (n_samples, 110)
y = np.load('data/key_output.npy')  # shape: (n_samples, 235)

# Optional: log compress features if they’re highly skewed
X = np.log1p(X)

# 2. Binary-ify output (multi-label)
y_binary = (y > 1e-5).astype(np.uint8)

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 4. Define Random Forest model (wrapped in MultiOutputClassifier)
base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(base_rf)

# 5. Fit model
model.fit(X_train, y_train)

# 6. Predict on test set
y_pred = model.predict(X_test)

# 7. Evaluate
precision = precision_score(y_test, y_pred, average='samples', zero_division=0)
recall = recall_score(y_test, y_pred, average='samples', zero_division=0)
f1 = f1_score(y_test, y_pred, average='samples', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
