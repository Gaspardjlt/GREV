import numpy as np
import pandas as pd

# Chargement du CSV
metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
print(metadata.shape)

# 1️⃣ Supprimer colonnes inutiles
cols_to_drop = ["Organism group", "ID", "Create date"]
metadata = metadata.drop(columns=[c for c in cols_to_drop if c in metadata.columns])

# 2️⃣ Supprimer toutes les colonnes "Unnamed"
metadata = metadata.loc[:, ~metadata.columns.str.contains("^Unnamed")]

# Colonnes qualitatives à traiter
cols_categorical = [
    "Isolation type",
    "Location",
    "Isolation source",
    "Laboratory typing platform",
    "Testing standard"
]

# 3️⃣ Gérer les valeurs manquantes
metadata[cols_categorical] = metadata[cols_categorical].fillna("Missing")

# 4️⃣ Garder uniquement les 2 valeurs les plus fréquentes par colonne
metadata_processed = metadata.copy()

for col in cols_categorical:
    top_values = metadata_processed[col].value_counts().nlargest(2).index
    metadata_processed[col] = metadata_processed[col].apply(lambda x: x if x in top_values else "Other")


# 5️⃣ One-Hot Encoding
metadata_ohe = pd.get_dummies(metadata_processed, columns=cols_categorical, drop_first=False)

print(metadata_ohe.columns)
print("Shape after one-hot:", metadata_ohe.shape)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # ou autre métrique selon ton problème

# Supposons que metadata_features contient tes 8 features
# et target contient la variable à prédire

data = np.load("inf-8245-fall-2025/train.npz")
y = data["y_train"]

X = metadata_ohe

# 1️⃣ Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2️⃣ Convertir en DMatrix (optionnel mais recommandé pour XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 3️⃣ Définir les paramètres du modèle
params = {
    'objective': 'binary:logistic',  # changer en 'reg:squarederror' si regression
    'max_depth': 4,
    'eta': 0.1,
    'eval_metric': 'logloss'
}

# 4️⃣ Entraîner le modèle
bst = xgb.train(params, dtrain, num_boost_round=50, evals=[(dtest, 'test')])

# 5️⃣ Prédire sur le test
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)  # pour classification binaire

# 6️⃣ Évaluer
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

