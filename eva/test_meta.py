import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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



