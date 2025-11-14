import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------------------
# 1) Charger données
# ------------------------------------
data = np.load("inf-8245-fall-2025/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

print(f"Original shape : {X_train.shape}")

# ------------------------------------
# 2) Réduire préliminaire à 100 000 colonnes
# ------------------------------------
MAX_COLS = 100_000
X_train = X_train[:, :MAX_COLS]
print(f"Shape après réduction préliminaire : {X_train.shape}")

# ------------------------------------
# 3) Supprimer colonnes constantes
# ------------------------------------
col_std = X_train.std(axis=0)
non_constant_cols = np.where(col_std > 1e-12)[0]
X_train = X_train[:, non_constant_cols]
print(f"Colonnes non constantes : {X_train.shape[1]}")

# ------------------------------------
# 4) Sélection des top 20 000 features par variance
# ------------------------------------
variances = X_train.var(axis=0)
top_idx = np.argsort(variances)[-20000:]   # top 20k
X_train = X_train[:, top_idx]
print(f"Shape après top 20k features : {X_train.shape}")

# ------------------------------------
# 5) Standardisation
# ------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ------------------------------------
# 6) Train/val
# ------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.15, random_state=42
)

# ------------------------------------
# 7) XGBoost optimisé pour dataset petit & large
# ------------------------------------
model = xgb.XGBClassifier(
    tree_method="hist",
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.6,
    colsample_bytree=0.2,
    reg_alpha=2,
    reg_lambda=3,
    eval_metric="logloss"
)

print("⏳ Entraînement...")
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# ------------------------------------
# 8) Préparer le test avec mêmes features
# ------------------------------------
data_test = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
X_test_full = data_test["X_test"]
ids_test = data_test["ids"]

# appliquer mêmes colonnes : non_constant_cols puis top_idx
X_test_selected = X_test_full[:, :MAX_COLS]
X_test_selected = X_test_selected[:, non_constant_cols]
X_test_selected = X_test_selected[:, top_idx]

# standardisation
X_test_scaled = scaler.transform(X_test_selected)

# prédiction
print("⏳ Prédiction test...")
y_pred = model.predict(X_test_scaled)

# ------------------------------------
# 9) Sauvegarde submission
# ------------------------------------
submission = pd.DataFrame({"id": ids_test, "label": y_pred.astype(int)})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv généré !")
