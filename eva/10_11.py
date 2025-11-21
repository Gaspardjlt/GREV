import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------
# 1) Charger train complet en RAM
# -------------------------------
data = np.load("inf-8245-fall-2025/train.npz")
X_train = data['X_train']
y_train = data['y_train']
ids_train = data['ids']

print(f"✅ X_train shape: {X_train.shape}")

# -------------------------------
# 2) Supprimer les colonnes constantes
# -------------------------------
std = X_train.std(axis=0)
non_constant_cols = np.where(std > 0)[0]

X_train = X_train[:, non_constant_cols]
print(f"✅ Colonnes conservées: {X_train.shape[1]}")

# -------------------------------
# 3) Standardisation train
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------
# 4) Train/Validation split
# -------------------------------
X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train_scaled, y_train, test_size=0.15, random_state=42
)

# -------------------------------
# 5) XGBoost
# -------------------------------
tree_method = "gpu_hist"
try:
    _ = xgb.DeviceQuantileDMatrix(X_train_split, y_train_split)
except:
    tree_method = "hist"

model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_lambda=1.5,
    reg_alpha=0.5,
    tree_method=tree_method,
    eval_metric="logloss",
    random_state=42
)

print("⏳ Entraînement du modèle...")
model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=100,
    verbose=50
)
print("✅ Modèle entraîné")

# -------------------------------
# 6) Prédiction X_test par chunks
# -------------------------------
data_test = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
ids_test = data_test['ids']

chunk_size = 10000
n_rows = data_test['X_test'].shape[0]
predictions = []

print(f"⏳ Prédiction sur {n_rows} lignes par chunks de {chunk_size}...")

for start in tqdm(range(0, n_rows, chunk_size)):
    end = min(start + chunk_size, n_rows)
    
    # lecture chunk
    X_chunk = data_test['X_test'][start:end, :]
    
    # suppression colonnes constantes
    X_chunk = X_chunk[:, non_constant_cols]
    
    # standardisation
    X_chunk_scaled = scaler.transform(X_chunk)
    
    # prédiction
    y_chunk_pred = model.predict(X_chunk_scaled)
    
    predictions.append(y_chunk_pred)

# concaténer toutes les prédictions
y_pred = np.concatenate(predictions)

# -------------------------------
# 7) Création submission.csv
# -------------------------------
submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv généré !")
