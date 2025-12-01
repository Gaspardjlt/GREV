import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION ET CHARGEMENT DES FICHIERS INPUT
# ==============================================================================

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
n_features = data['X_train'].shape[1]
CHUNK_SIZE = 100000

print("-" * 50)

# ==============================================================================
# 1. PRÉ-FILTRAGE LÉGER PAR VARIANCE (CONSERVÉ)
# ==============================================================================
print("1. Pré-filtrage par Variance pour la sécurité RAM...")

all_feature_variances = []
for i in tqdm(range(0, n_features, CHUNK_SIZE), desc="Calcul de Variance"):
    chunk_start = i
    chunk_end = min(i + CHUNK_SIZE, n_features)
    current_chunk = data['X_train'][:, chunk_start:chunk_end] 
    current_variances = np.var(current_chunk, axis=0)
    all_feature_variances.append(current_variances)
    del current_chunk

combined_variances = np.concatenate(all_feature_variances)
SAFE_VARIANCE_THRESHOLD = 0.045
safe_indices = np.where(combined_variances > SAFE_VARIANCE_THRESHOLD)[0]

X_train_reduced = data['X_train'][:, safe_indices]
X_test_reduced = data_test['X_test'][:, safe_indices]

print(f"Features conservées après le pré-filtrage (taille gérable) : {X_train_reduced.shape[1]:,}")
print("-" * 50)


# ==============================================================================
# 2. STANDARDISATION ET CLASSEMENT INITIAL DES FEATURES
# ==============================================================================
print("2. Standardisation et Classement des features (un seul entraînement Ridge)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)
del X_train_reduced, X_test_reduced # Libération mémoire

# --- ÉTAPE CLÉ : Entraînement UNIQUE d'un modèle Ridge pour CLASSER les features ---
print(" -> Entraînement initial d'un modèle Ridge sur toutes les features standardisées...")

# On utilise un modèle Ridge pour le classement (peu importe l'alpha ici)
ridge_ranker = SGDClassifier(
    loss='log_loss', penalty='l2', alpha=1.0, random_state=42, max_iter=1000, tol=1e-3, n_jobs=-1
)
ridge_ranker.fit(X_train_scaled, y_train)

# Classement des indices par l'importance (valeur absolue des coefficients)
# top_indices_ranked contient les indices de X_train_scaled/X_test_scaled
top_indices_ranked = np.argsort(np.abs(ridge_ranker.coef_[0]))[::-1]
print("Classement des features terminé.")
print("-" * 50)


# ==============================================================================
# 3. DOUBLE OPTIMISATION ALPHA/K (Grid Search Manuelle)
# ==============================================================================
print("3. Double Optimisation Alpha/K par Grid Search Manuelle...")

# Plages de valeurs à tester
alpha_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0] 
k_values = np.arange(1500, 6501, 500) # De 1500 à 6500

best_global_score = -1.0
best_global_k = 0
best_global_alpha = 0.0
best_global_indices = None

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for K in tqdm(k_values, desc="Optimisation K"):
    # Sélectionner les K meilleures features basées sur le classement unique
    current_top_indices = top_indices_ranked[:K]
    X_train_current_K = X_train_scaled[:, current_top_indices]
    
    # Grid Search pour Alpha (L2) sur les K features sélectionnées
    ridge_base = SGDClassifier(
        loss='log_loss', penalty='l2', random_state=42, max_iter=1000, tol=1e-3, n_jobs=-1
    )
    
    param_grid = {'alpha': alpha_values} 
    grid_search = GridSearchCV(
        estimator=ridge_base, param_grid=param_grid, scoring='roc_auc', cv=cv, verbose=0, n_jobs=-1 
    )
    
    grid_search.fit(X_train_current_K, y_train)
    current_best_score = grid_search.best_score_
    
    if current_best_score > best_global_score:
        best_global_score = current_best_score
        best_global_k = K
        best_global_alpha = grid_search.best_params_['alpha']
        best_global_indices = current_top_indices

print(f"\n##################################################")
print(f"✅ MEILLEURS PARAMÈTRES GLOBAUX TROUVÉS:")
print(f"   - K (Features) : {best_global_k:,}")
print(f"   - Alpha (Ridge): {best_global_alpha}")
print(f"   - AUC (CV)     : {best_global_score:.6f}")
print("##################################################")
print("-" * 50)


# ==============================================================================
# 4. SÉLECTION FINALE ET ENTRAÎNEMENT XGBOOST
# ==============================================================================
print(f"4. Sélection finale des {best_global_k:,} features et Entraînement XGBoost...")

# Création des matrices finales avec le meilleur K et les données standardisées
X_train_final = X_train_scaled[:, best_global_indices]
X_test_final = X_test_scaled[:, best_global_indices]

# Libération de la mémoire
del X_train_scaled, X_test_scaled 

# Entraînement final XGBoost
XGB_REG_LAMBDA = 0.1
XGB_REG_ALPHA = 0.1

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.5, min_child_weight=3, 
    reg_lambda=XGB_REG_LAMBDA, reg_alpha=XGB_REG_ALPHA,
    tree_method='hist', n_jobs=-1, random_state=42, verbosity=0
)

xgb.fit(X_train_final, y_train)

print(f"Modèle XGBoost entraîné sur le TOP {best_global_k:,} features.")
print("-" * 50)


# ==============================================================================
# 5. PRÉDICTION ET FICHIER DE SOUMISSION
# ==============================================================================
y_pred_proba = xgb.predict_proba(X_test_final)[:, 1] 

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred_proba 
})

submission_filename = f"submission_final_k{best_global_k}_alpha{best_global_alpha:.4f}_xgb.csv".replace(".", "_")
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")