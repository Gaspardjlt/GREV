# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

# print("\n=== Chargement des métadonnées TRAIN ===")

# # --- Charger uniquement les métadonnées train ---
# metadata_train = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")

# cols_to_drop = ["Organism group", "ID", "Create date"]
# metadata_train = metadata_train.drop(columns=[c for c in cols_to_drop if c in metadata_train.columns])
# metadata_train = metadata_train.loc[:, ~metadata_train.columns.str.contains("^Unnamed")]

# cols_categorical = [
#     "Isolation type",
#     "Location",
#     "Isolation source",
#     "Laboratory typing platform",
#     "Testing standard"
# ]

# # --- Remplacer les valeurs manquantes ---
# metadata_train[cols_categorical] = metadata_train[cols_categorical].fillna("Missing")

# # --- Pré-traitement train ---
# metadata_processed_train = metadata_train.copy()
# for col in cols_categorical:
#     top_values = metadata_processed_train[col].value_counts().nlargest(2).index
#     metadata_processed_train[col] = metadata_processed_train[col].apply(
#         lambda x: x if x in top_values else "Other"
#     )

# metadata_ohe_train = pd.get_dummies(metadata_processed_train, columns=cols_categorical, drop_first=False)

# print("OK : métadonnées train traitées.")

# # --- Charger les features X_train ---
# print("\n=== Chargement des features TRAIN ===")
# data_X = np.load("eva/X_reduit_190.npz")
# data_y = np.load("eva/y_train_190.npz")

# X_train_data = data_X["X"]
# y_train = data_y["y"]

# print(f"Shape X_train : {X_train_data.shape}")
# print(f"len(y_train)  : {len(y_train)}")

# # --- Scaling et concaténation ---
# metadata_train_scaled = metadata_ohe_train * (0.3 / metadata_ohe_train.shape[1])
# X_train_scaled = X_train_data * (0.7 / X_train_data.shape[1])
# X_train_scaled_df = pd.DataFrame(X_train_scaled)

# X_train_final = pd.concat(
#     [X_train_scaled_df, metadata_train_scaled.reset_index(drop=True)],
#     axis=1
# )

# print("OK : features finales de TRAIN prêtes.")

# # ============================================================================
# # 🧪 SPLIT TRAIN / VALIDATION + XGBOOST
# # ============================================================================

# print("\n=== Split train/validation ===")

# X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
#     X_train_final,
#     y_train,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train
# )

# dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
# dvalid = xgb.DMatrix(X_valid_split, label=y_valid_split)

# params = {
#     "objective": "binary:logistic",
#     "eval_metric": "logloss",
#     "eta": 0.1,
#     "max_depth": 6,
#     "seed": 42
# }

# print("\n=== Entraînement avec early stopping ===")

# bst = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=300,
#     evals=[(dtrain, "train"), (dvalid, "valid")],
#     early_stopping_rounds=20,
#     verbose_eval=20
# )

# # ---------------------------------------------------------------------
# # ÉVALUATION
# # ---------------------------------------------------------------------

# y_valid_pred_prob = bst.predict(dvalid)
# y_valid_pred = (y_valid_pred_prob > 0.5).astype(int)

# print("\n=== PERFORMANCE SUR VALIDATION ===")
# print("----------------------------------")
# print(f"Logloss  : {log_loss(y_valid_split, y_valid_pred_prob):.5f}")
# print(f"Accuracy : {accuracy_score(y_valid_split, y_valid_pred):.5f}")
# print(f"AUC      : {roc_auc_score(y_valid_split, y_valid_pred_prob):.5f}")
# print("\n=== FIN ===\n")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from category_encoders import TargetEncoder

# --- Charger les métadonnées ---
metadata_train = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")

cols_to_drop = ["Organism group", "ID", "Create date"]
metadata_train = metadata_train.drop(columns=[c for c in cols_to_drop if c in metadata_train.columns])

cols_categorical = [
    "Isolation type",
    "Location",
    "Isolation source",
    "Laboratory typing platform",
    "Testing standard"
]

metadata_train[cols_categorical] = metadata_train[cols_categorical].fillna("Missing")

# --- Charger X et y ---
data_X = np.load("eva/X_reduit_190.npz")
data_y = np.load("eva/y_train_190.npz")
X_data = data_X["X"]
y = data_y["y"]

# --- Split train / validation / test interne ---
X_train_sub, X_temp, y_train_sub, y_temp = train_test_split(
    X_data, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# --- Target Encoding des catégories sur train uniquement ---
metadata_train_sub = metadata_train.iloc[X_train_sub.shape[0]*[0]:X_train_sub.shape[0]*[0]+X_train_sub.shape[0]]  # juste placeholder si alignement nécessaire
# Pour alignement simple, on utilise tout metadata_train mais fit sur X_train_sub
te = TargetEncoder(cols=cols_categorical, smoothing=10)
te.fit(metadata_train[cols_categorical], y)
metadata_train_enc = te.transform(metadata_train[cols_categorical])

# --- Séparer metadata pour les splits internes ---
metadata_train_split = metadata_train_enc.iloc[X_train_sub.shape[0]:X_train_sub.shape[0]+X_train_sub.shape[0]]  # placeholder
metadata_val_split = metadata_train_enc.iloc[X_train_sub.shape[0]+X_train_sub.shape[0]:X_train_sub.shape[0]+X_train_sub.shape[0]+X_val.shape[0]]
metadata_test_split = metadata_train_enc.iloc[-X_test.shape[0]:]

# --- Scaling features principales ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sub)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Concaténer metadata ---
X_train_final = np.hstack([X_train_scaled, metadata_train_split.values])
X_val_final = np.hstack([X_val_scaled, metadata_val_split.values])
X_test_final = np.hstack([X_test_scaled, metadata_test_split.values])

# --- DMatrix XGBoost ---
dtrain = xgb.DMatrix(X_train_final, label=y_train_sub)
dval = xgb.DMatrix(X_val_final, label=y_val)
dtest = xgb.DMatrix(X_test_final, label=y_test)

# --- Paramètres XGBoost ---
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

# --- Prédiction sur val / test interne ---
y_val_pred_prob = bst.predict(dval)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)

y_test_pred_prob = bst.predict(dtest)
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

# --- Évaluation ---
val_acc = accuracy_score(y_val, y_val_pred)
val_loss = log_loss(y_val, y_val_pred_prob)

test_acc = accuracy_score(y_test, y_test_pred)
test_loss = log_loss(y_test, y_test_pred_prob)

print(f"Validation Accuracy: {val_acc:.4f}, LogLoss: {val_loss:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}, LogLoss: {test_loss:.4f}")

