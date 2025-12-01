import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
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

CHUNK_SIZE = 100000
FINAL_K_FIXED = 6000 # Nombre de features fixé

print(f"Dimensions cibles : ({n_samples}, {n_features:,})")
print("-" * 50)


# ==============================================================================
# 1. PRÉ-FILTRAGE LÉGER PAR VARIANCE (Sécurité Anti-RAM)
# ==============================================================================
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

safe_indices = np.where(combined_variances > 0.045)[0]
# Chargement des données réduites en RAM
X_train_reduced = data['X_train'][:, safe_indices]
X_test_reduced = data_test['X_test'][:, safe_indices]

print(f"Features conservées après le pré-filtrage (taille gérable) : {X_train_reduced.shape[1]:,}")
print("-" * 50)


# ==============================================================================
# 2. STANDARDISATION ET OPTIMISATION ALPHA RIDGE (L2)
# ==============================================================================
print("2. Standardisation et Optimisation de Alpha (Grid Search L2)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Séparation des données pour la validation XGBoost (utilisée plus tard)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# --- GRID SEARCH pour Alpha ---
print("  -> Recherche du meilleur Alpha (Ridge) par Cross-Validation...")

# Plage de valeurs d'alpha à tester
param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0]} # Teste des puissances de 10

# Configuration du modèle Ridge
ridge_base = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    random_state=42, max_iter=1000, tol=1e-3,
    n_jobs=1
)

# Configuration de la Grid Search avec StratifiedKFold pour la robustesse
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=ridge_base,
    param_grid=param_grid,
    scoring='roc_auc', # Utilise l'AUC comme métrique d'évaluation
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_alpha = grid_search.best_params_['alpha']
best_score = grid_search.best_score_

print(f"✅ MEILLEUR ALPHA TROUVÉ: {best_alpha} avec AUC (CV) de {best_score:.6f}")
# --- FIN GRID SEARCH pour Alpha ---

# Entraînement du Modèle Ridge final avec le meilleur alpha
ridge_model = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=best_alpha, # Utilisation du meilleur alpha
    random_state=42, max_iter=1000, tol=1e-3,
    n_jobs=-1 # Peut utiliser tous les cœurs pour l'entraînement final
)

ridge_model.fit(X_train_scaled, y_train)

# Récupération et tri des indices par importance (valeur absolue des coefficients)
top_indices_ridge = np.argsort(np.abs(ridge_model.coef_[0]))[::-1]
print("Modèle Ridge entraîné avec alpha optimisé. Coefficients prêts pour le classement.")
print("-" * 50)


# ==============================================================================
# 3. SÉLECTION FINALE À 6000 FEATURES (Valeur fixée)
# ==============================================================================
print(f"3. Sélection des {FINAL_K_FIXED:,} features (Valeur fixée)...")

# Création des matrices d'entraînement et de validation avec 6000 features
final_top_indices = top_indices_ridge[:FINAL_K_FIXED]
X_train_final = X_train_scaled[:, final_top_indices]
X_test_final = X_test_scaled[:, final_top_indices]

# Libération de la mémoire des matrices intermédiaires
del X_train_reduced, X_test_reduced, X_train_scaled, X_test_scaled

# Les splits pour la validation XGBoost (X_tr, X_val) ne sont plus nécessaires car
# l'optimisation des hyperparamètres XGBoost est omise cette fois, mais nous les gardons
# pour le délétion en toute sécurité si la structure d'optimisation devait revenir.

print(f"Dimensions finales (X_train) : {X_train_final.shape}")
print("-" * 50)


# ==============================================================================
# 4. ENTRAÎNEMENT FINAL XGBOOST
# ==============================================================================
# Utilisation des meilleurs paramètres de régularisation trouvés précédemment (0.1 et 0.1)
# ou de valeurs par défaut si aucune recherche n'a été faite, pour rester cohérent.
XGB_REG_LAMBDA = 0.1
XGB_REG_ALPHA = 0.1

print(f"4. Entraînement final du modèle XGBoost sur les {FINAL_K_FIXED:,} features...")
print(f"   -> Utilisation des paramètres de régularisation : lambda={XGB_REG_LAMBDA}, alpha={XGB_REG_ALPHA}")

# Fermeture des fichiers npz après l'extraction des données nécessaires
data.close()
data_test.close()

# Entraînement final (300 estimateurs)
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

print(f"Modèle XGBoost entraîné sur le TOP {FINAL_K_FIXED:,} features.")
print("-" * 50)


# ==============================================================================
# 5. PRÉDICTION ET FICHIER DE SOUMISSION
# ==============================================================================
y_pred = xgb.predict(X_test_final)

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = f"submission_final_k{FINAL_K_FIXED}_alpha{best_alpha:.4f}_xgb.csv".replace(".", "_")
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")