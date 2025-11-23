import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier # Utilisé pour la sélection L2
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION ET CHARGEMENT DES FICHIERS INPUT (Chemins GREV)
# ==============================================================================

# VÉRIFIEZ ET AJUSTEZ CE CHEMIN
BASE_PATH = "inf-8245-fall-2025/" 
TRAIN_PATH = os.path.join(BASE_PATH, "train.npz")
TEST_PATH = os.path.join(BASE_PATH, "test.npz")

print(f"0. Chargement des données depuis : {BASE_PATH} (Mode Anti-RAM)...")
try:
    data = np.load(TRAIN_PATH)
    data_test = np.load(TEST_PATH)
except FileNotFoundError:
    print(f"ERREUR: Fichiers non trouvés au chemin: {TRAIN_PATH}. Veuillez vérifier le chemin d'accès.")
    exit()

y_train = data['y_train']
ids_test = data_test['ids']
n_samples = y_train.shape[0]
n_features = data['X_train'].shape[1]

final_k = 5000
CHUNK_SIZE = 100000 

print(f"Dimensions cibles : ({n_samples}, {n_features:,})")
print("-" * 50)


# ==============================================================================
# 1. PRÉ-FILTRAGE LÉGER PAR VARIANCE (Sécurité Anti-RAM)
# ==============================================================================
# Nous devons toujours faire un pré-filtrage par variance pour retirer les features mortes
# et réduire la matrice à une taille gérable (ex: 477k) AVANT d'entraîner Ridge.
print("1. Pré-filtrage par Variance pour la sécurité RAM...")

all_feature_variances = []

for i in tqdm(range(0, n_features, CHUNK_SIZE), desc="Pré-filtrage par Variance"):
    chunk_start = i
    chunk_end = min(i + CHUNK_SIZE, n_features)
    current_chunk = data['X_train'][:, chunk_start:chunk_end]
    current_variances = np.var(current_chunk, axis=0) 
    all_feature_variances.append(current_variances)
    del current_chunk 

combined_variances = np.concatenate(all_feature_variances)

# On garde toutes les features au-dessus du seuil 0.0001
safe_indices = np.where(combined_variances > 0.02)[0]
X_train_reduced = data['X_train'][:, safe_indices]
X_test_reduced = data_test['X_test'][:, safe_indices]

print(f"Features conservées après le pré-filtrage (taille gérable) : {X_train_reduced.shape[1]:,}")
print("-" * 50)


# ==============================================================================
# 2. STANDARDISATION ET ENTRAÎNEMENT RIDGE (L2)
# ==============================================================================
print("2. Standardisation et Entraînement du Modèle L2 (Ridge) pour le scoring...")

# Standardisation de la matrice réduite
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Modèle Ridge (SGDClassifier avec penalty='l2') - Pas de CV, un seul entraînement
ridge_model = SGDClassifier(
    loss='log_loss',         # Utilise la régression logistique
    penalty='l2',            # Régularisation L2
    alpha=0.01,              # Force de régularisation fixée
    random_state=42, max_iter=1000, tol=1e-3,
    n_jobs=1                 # IMPORTANT: Mode séquentiel pour la RAM
)

ridge_model.fit(X_train_scaled, y_train)

print("Modèle Ridge entraîné. Utilisation des coefficients pour le classement.")
print("-" * 50)


# ==============================================================================
# 3. SÉLECTION FINALE DU TOP 5000 PAR COEFFICIENTS RIDGE
# ==============================================================================
print(f"3. Sélection finale du TOP {final_k:,} features via Coefficients L2...")

# Récupération de la valeur absolue des coefficients Ridge
# Les coefficients plus grands indiquent une plus grande importance pour la classification
coefs = np.abs(ridge_model.coef_[0])

# Sélection des indices des 5000 plus grands coefficients
top_indices_final = np.argsort(coefs)[::-1][:final_k]

# Création des matrices finales
X_train_final = X_train_scaled[:, top_indices_final]
X_test_final = X_test_scaled[:, top_indices_final]

# Libération de la mémoire
del X_train_reduced, X_test_reduced, X_train_scaled, X_test_scaled
data.close()
data_test.close()

print(f"Dimensions finales (X_train) : {X_train_final.shape}")
print("-" * 50)


# ==============================================================================
# 4. ENTRAÎNEMENT XGBOOST AVEC SUPPORT GPU
# ==============================================================================
print("4. Entraînement du modèle XGBoost...")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=3,
    reg_lambda=1.0,
    tree_method='hist', 
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

xgb.fit(X_train_final, y_train)

print("Modèle XGBoost entraîné sur le TOP 5000 features par score L2.")
print("-" * 50)


# ==============================================================================
# 5. PRÉDICTION ET FICHIER DE SOUMISSION
# ==============================================================================
y_pred = xgb.predict(X_test_final)

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = "submission_final_top_5000_L2.csv"
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")