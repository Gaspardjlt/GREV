import numpy as np
import pandas as pd
import xgboost as xgb

# --- Charger les métadonnées ---
# --- Charger les métadonnées ---
metadata_train = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
metadata_test = pd.read_csv("inf-8245-fall-2025/metadata_test.csv")

cols_to_drop = ["Organism group", "ID", "Create date"]

metadata_train = metadata_train.drop(columns=[c for c in cols_to_drop if c in metadata_train.columns])
metadata_train = metadata_train.loc[:, ~metadata_train.columns.str.contains("^Unnamed")]

metadata_test = metadata_test.drop(columns=[c for c in cols_to_drop if c in metadata_test.columns])
metadata_test = metadata_test.loc[:, ~metadata_test.columns.str.contains("^Unnamed")]

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

# --- Pré-traitement train ---
metadata_processed_train = metadata_train.copy()
for col in cols_categorical:
    top_values = metadata_processed_train[col].value_counts().nlargest(2).index
    metadata_processed_train[col] = metadata_processed_train[col].apply(lambda x: x if x in top_values else "Other")

metadata_ohe_train = pd.get_dummies(metadata_processed_train, columns=cols_categorical, drop_first=False)

# --- Pré-traitement test ---
metadata_processed_test = metadata_test.copy()
for col in cols_categorical:
    # ⚠ Ne PAS reprendre les top valeurs du test !
    # On doit appliquer la logique du TRAIN pour garantir les mêmes catégories.
    top_values = metadata_processed_train[col].value_counts().nlargest(2).index
    metadata_processed_test[col] = metadata_processed_test[col].apply(lambda x: x if x in top_values else "Other")

metadata_ohe_test = pd.get_dummies(metadata_processed_test, columns=cols_categorical, drop_first=False)

# --- Alignement des colonnes (très important !) ---
metadata_ohe_train, metadata_ohe_test = metadata_ohe_train.align(
    metadata_ohe_test, join="outer", axis=1, fill_value=0
)

# --- Charger les features principales ---
data_X = np.load("eva/X_reduit_190.npz")
data_y = np.load("eva/y_train_190.npz")
# print(data.files)
X_train_data = data_X["X"]
y_train = data_y["y"]

data_test = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
data_test_X = np.load("inf-8245-fall-2025/X_test_reduit_190.npz") 
X_test_data = data_test_X["X"]
ids_test = data_test["ids"]

# --- Vérifier dimensions et ajuster metadata pour train/test ---
# Si les métadonnées sont les mêmes pour train et test, sinon il faudra faire merge sur ID
metadata_train_scaled = metadata_ohe_train * (0.3 / metadata_ohe_train.shape[1])
X_train_scaled = X_train_data * (0.7 / X_train_data.shape[1])
# Si X_train_scaled est un numpy array
X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_train_final = pd.concat([X_train_scaled_df, metadata_train_scaled.reset_index(drop=True)], axis=1)

metadata_test_scaled = metadata_ohe_test * (0.3 / metadata_ohe_test.shape[1])
X_test_scaled = X_test_data * (0.7 / X_test_data.shape[1])
X_test_scaled_df = pd.DataFrame(X_test_scaled)
X_test_final = pd.concat([X_test_scaled_df, metadata_test_scaled.reset_index(drop=True)], axis=1)

# --- Créer DMatrix pour XGBoost ---
dtrain = xgb.DMatrix(X_train_final, label=y_train)
dtest = xgb.DMatrix(X_test_final)

# --- Paramètres XGBoost ---
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.1,
    "max_depth": 6,
    "seed": 42
}

# --- Entraînement ---
bst = xgb.train(params, dtrain, num_boost_round=100)

# --- Prédiction ---
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# --- Submission Kaggle ---
submission = pd.DataFrame({"id": ids_test, "label": y_pred})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv généré !")
