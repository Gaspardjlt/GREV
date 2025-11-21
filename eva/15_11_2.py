import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# --- Charger les métadonnées ---
metadata_train = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
metadata_test = pd.read_csv("inf-8245-fall-2025/metadata_test.csv")

cols_to_drop = ["Organism group", "ID", "Create date"]
metadata_train = metadata_train.drop(columns=[c for c in cols_to_drop if c in metadata_train.columns])
metadata_test = metadata_test.drop(columns=[c for c in cols_to_drop if c in metadata_test.columns])

cols_categorical = [
    "Isolation type",
    "Location",
    "Isolation source",
    "Laboratory typing platform",
    "Testing standard"
]

# --- Remplacer les valeurs manquantes ---
metadata_train[cols_categorical] = metadata_train[cols_categorical].fillna("Missing")
metadata_test[cols_categorical] = metadata_test[cols_categorical].fillna("Missing")

# --- Charger X et y ---
data_X = np.load("eva/X_reduit_190.npz")
data_y = np.load("eva/y_train_190.npz")
X_train_data = data_X["X"]
y_train = data_y["y"]

data_test_X = np.load("inf-8245-fall-2025/X_test_reduit_190.npz")
X_test_data = data_test_X["X"]

# --- Target Encoding des catégories ---
te = TargetEncoder(cols=cols_categorical, smoothing=10)
te.fit(metadata_train[cols_categorical], y_train)

metadata_train_enc = te.transform(metadata_train[cols_categorical])
metadata_test_enc = te.transform(metadata_test[cols_categorical])

# --- Scaling features principales ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_data)
X_test_scaled = scaler.transform(X_test_data)

# --- Concaténer metadata + features principales ---
X_train_final = np.hstack([X_train_scaled, metadata_train_enc.values])
X_test_final = np.hstack([X_test_scaled, metadata_test_enc.values])

# --- Split validation ---
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test_final)

# --- Paramètres XGBoost optimisés ---
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.02,
    "max_depth": 7,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "lambda": 2.0,
    "alpha": 0.2,
    "min_child_weight": 5,
    "tree_method": "hist",
    "seed": 42
}

# --- Entraînement avec early stopping ---
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=80,
    verbose_eval=100
)

# --- Prédiction ---
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Submission Kaggle ---
data_test = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
ids_test = data_test["ids"]

submission = pd.DataFrame({"id": ids_test, "label": y_pred})
submission.to_csv("submission_optimized_no_date.csv", index=False)
print("✅ submission_optimized_no_date.csv généré !")
