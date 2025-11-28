import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
print("Original shape:", metadata.shape)

# 1️⃣ Supprimer colonnes inutiles
cols_to_drop = ["Organism group", "ID", "Create date"]
metadata = metadata.drop(columns=[c for c in cols_to_drop if c in metadata.columns])

# 2️⃣ Supprimer toutes les colonnes "Unnamed"
metadata = metadata.loc[:, ~metadata.columns.str.contains("^Unnamed")]

# 3️⃣ Gérer les valeurs manquantes pour toutes les colonnes
for col in metadata.columns:
    if metadata[col].dtype == object:
        metadata[col] = metadata[col].fillna("Missing")
    else:
        metadata[col] = metadata[col].fillna(metadata[col].median())

# 4️⃣ One-Hot Encoding sur toutes les colonnes catégorielles
metadata_ohe = pd.get_dummies(metadata, drop_first=False)
print("Shape after one-hot:", metadata_ohe.shape)

# 5️⃣ Séparer features et target
# ⚠️ Remplacer 'Target' par le nom réel de ta colonne cible
data = np.load("inf-8245-fall-2025/train.npz")
# X = data['X_train']
y = data['y_train']
# target_col = "Target"
X = metadata_ohe
# y = metadata_ohe[target_col]

# 6️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Entraîner XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Prédictions et accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9️⃣ Importance prédictive des features
importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(importances.head(20))
