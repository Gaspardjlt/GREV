import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1) CHARGER MÉTADATA TRAIN
# -----------------------------
metadata_train = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
metadata_test = pd.read_csv("inf-8245-fall-2025/metadata_test.csv")

cols_to_drop = ["Organism group", "ID", "Create date"]
metadata_train = metadata_train.drop(columns=[c for c in cols_to_drop if c in metadata_train.columns])
metadata_test = metadata_test.drop(columns=[c for c in cols_to_drop if c in metadata_test.columns])

# supprimer colonnes Unnamed
metadata_train = metadata_train.loc[:, ~metadata_train.columns.str.contains("^Unnamed")]
metadata_test  = metadata_test.loc[:,  ~metadata_test.columns.str.contains("^Unnamed")]

cols_categorical = [
    "Isolation type",
    "Location",
    "Isolation source",
    "Laboratory typing platform",
    "Testing standard"
]

# Remplacer NA
metadata_train[cols_categorical] = metadata_train[cols_categorical].fillna("Missing")
metadata_test[cols_categorical]  = metadata_test[cols_categorical].fillna("Missing")

# -----------------------------
# 2) TOP-2 catégories pour train ET test
# -----------------------------
metadata_train_processed = metadata_train.copy()
metadata_test_processed  = metadata_test.copy()

for col in cols_categorical:
    top_values = metadata_train_processed[col].value_counts().nlargest(2).index
    
    metadata_train_processed[col] = metadata_train_processed[col].apply(
        lambda x: x if x in top_values else "Other"
    )
    metadata_test_processed[col] = metadata_test_processed[col].apply(
        lambda x: x if x in top_values else "Other"
    )

# -----------------------------
# 3) ONE-HOT ENCODING
# -----------------------------
metadata_train_ohe = pd.get_dummies(metadata_train_processed, columns=cols_categorical, drop_first=False)
metadata_test_ohe  = pd.get_dummies(metadata_test_processed,  columns=cols_categorical, drop_first=False)

# Aligner les colonnes train/test
metadata_test_ohe = metadata_test_ohe.reindex(columns=metadata_train_ohe.columns, fill_value=0)

print("Train shape:", metadata_train_ohe.shape)
print("Test shape :", metadata_test_ohe.shape)

# -----------------------------
# 4) CHARGER Y
# -----------------------------
data_y = np.load("eva/y_train_190.npz")
y_train = data_y["y"]

# -----------------------------
# 5) TRAIN/VAL SPLIT
# -----------------------------
X_train, X_val, y_train_sub, y_val = train_test_split(
    metadata_train_ohe, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# -----------------------------
# 6) XGBOOST MODEL
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=3,
    gamma=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train_sub)

# Évaluer
y_pred_val = model.predict(X_val)
acc = accuracy_score(y_val, y_pred_val)
print("\n🎯 Accuracy metadata only:", acc)

# -----------------------------
# 7) PRÉDICTION TEST METADATA
# -----------------------------
y_pred_test = model.predict(metadata_test_ohe)

# -----------------------------
# 8) CRÉATION DU CSV DE SUBMISSION
# -----------------------------
data_test_ids = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
ids_test = data_test_ids["ids"]

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred_test
})

submission.to_csv("submission_metadata_only.csv", index=False)

print("\n✅ submission_metadata_only.csv généré !")
