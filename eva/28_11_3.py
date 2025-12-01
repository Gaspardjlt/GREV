import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# =============================
# 1. Chargement des données
# =============================
data = np.load("inf-8245-fall-2025/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

data_test = np.load("inf-8245-fall-2025/test.npz")
X_test = data_test["X_test"]
ids_test = data_test["ids"]

# =============================
# 2. Pipeline SVD + scaler + SGD
# =============================
# On réduit à 10 000 composantes principales
n_components = 10000

clf = Pipeline([
    ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=30000,
        tol=1e-4,
        learning_rate="optimal",
        n_jobs=-1,
        random_state=42
    ))
])

# =============================
# 3. Entraînement
# =============================
print("✅ Entraînement du modèle...")
clf.fit(X_train, y_train)

# =============================
# 4. Prédiction
# =============================
y_pred = clf.predict(X_test)

# =============================
# 5. Création du CSV de soumission
# =============================
submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = "submission_sgd_svd.csv"
submission.to_csv(submission_filename, index=False)
print(f"✅ Fichier de soumission créé : {submission_filename}")
