import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------
# 1) Charger train
# -------------------------------
data = np.load("inf-8245-fall-2025/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

n_rows, n_cols = X_train.shape
print(f"Original shape : {X_train.shape}")

# -------------------------------
# 💡 2) Réduire à 100 000 colonnes
# -------------------------------
MAX_COLS = 100_000
X_train = X_train[:, :MAX_COLS]
print(f"Shape après réduction préliminaire : {X_train.shape}")

# -------------------------------
# 3) Détection colonnes constantes sur ces 100k
# -------------------------------
print("⏳ Détection colonnes constantes...")

first_val = X_train[0].copy()
is_const = np.ones(MAX_COLS, dtype=bool)

for i in range(1, n_rows):
    row = X_train[i]
    is_const[row != first_val] = False

non_constant_cols = np.where(~is_const)[0]
print(f"➡️ Colonnes non constantes : {len(non_constant_cols)}")

X_train = X_train[:, non_constant_cols]
print(f"Shape final train : {X_train.shape}")

# -------------------------------
# 4) Standardisation safe
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------
# 5) Train/val
# -------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.15, random_state=42
)

# -------------------------------
# 6) XGBoost
# -------------------------------
model = xgb.XGBClassifier(
    tree_method="hist", 
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

print("⏳ Entraînement...")
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

# -------------------------------
# 7) Test par chunks
# -------------------------------
data_test = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
X_test_full = data_test["X_test"]
ids_test = data_test["ids"]

preds = []
chunk_size = 10_000

print("⏳ Prédiction test...")

for start in tqdm(range(0, X_test_full.shape[0], chunk_size)):
    end = min(start + chunk_size, X_test_full.shape[0])
    chunk = X_test_full[start:end, :MAX_COLS]
    chunk = chunk[:, non_constant_cols]
    chunk = scaler.transform(chunk)
    preds.append(model.predict(chunk))

y_pred = np.concatenate(preds)

submission = pd.DataFrame({"id": ids_test, "label": y_pred.astype(int)})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv généré !")
