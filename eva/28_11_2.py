import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION ET CHARGEMENT DES FICHIERS INPUT (Chemins GREV)
# ==============================================================================

# VÉRIFIEZ ET AJUSTEZ CE CHEMIN vers votre dossier contenant train.npz et test.npz
data = np.load("inf-8245-fall-2025/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

data_test = np.load("inf-8245-fall-2025/test.npz")
X_test = data_test["X_test"]
ids_test = data_test['ids']

n_samples = y_train.shape[0]
n_features = data['X_train'].shape[1]

# exemple : prendre les colonnes correspondant aux 10k k-mers les plus fréquents
top_k_indices = np.argsort(np.sum(X_train, axis=0))[-600000:]
X_train = X_train[:, top_k_indices]
X_test  = X_test[:, top_k_indices]

# Permet de retirer les features dont la variance est 0 (k-mers jamais présents → courant !)
vt = VarianceThreshold(threshold=0.0)  # supprime features constantes
X_train_vt = vt.fit_transform(X_train)
X_test_vt = vt.transform(X_test)

from sklearn.feature_selection import SelectKBest, chi2

k = 10000
selector = SelectKBest(score_func=chi2, k=k)
X_train_sel = selector.fit_transform(X_train_vt, y_train)
X_test_sel = selector.transform(X_test_vt)

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('scaler', MaxAbsScaler()),  # obligatoire pour SGD, sparse friendly
    ('sgd', SGDClassifier(
        loss='log_loss',
        class_weight='balanced',
        max_iter=30_000,
        tol=1e-4,
        learning_rate='optimal',
        n_jobs=-1
    ))
])

clf.fit(X_train_sel, y_train)

y_pred = clf.predict(X_test_sel)



submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = "submission_28_11_2_600000.csv"
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")

# marche pas pour 750 000 et 650 000
