import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION ET CHARGEMENT DES FICHIERS INPUT (Chemins GREV)
# ==============================================================================

# VEUILLEZ VÉRIFIER ET AJUSTER CE CHEMIN SI NÉCESSAIRE.
# Exemple pour Colab avec Drive monté: /content/drive/MyDrive/GREV/...
BASE_PATH = "inf-8245-fall-2025/" 
TRAIN_PATH = os.path.join(BASE_PATH, "train.npz")
TEST_PATH = os.path.join(BASE_PATH, "test.npz")

print(f"0. Chargement des données depuis : {BASE_PATH} (Mode Anti-RAM)...")
try:
    # On ouvre les fichiers pour accéder aux données par indexation
    data = np.load(TRAIN_PATH)
    data_test = np.load(TEST_PATH)
except FileNotFoundError:
    print(f"ERREUR: Fichiers non trouvés au chemin: {TRAIN_PATH}. Vérifiez votre chemin d'accès.")
    exit()

# On charge uniquement les petits vecteurs (cibles et ids) en RAM
y_train = data['y_train']
ids_test = data_test['ids']
n_samples = y_train.shape[0]
n_features = data['X_train'].shape[1]

final_k = 5000
CHUNK_SIZE = 100000 

print(f"Dimensions cibles : ({n_samples}, {n_features:,})")
print("-" * 50)


# ==============================================================================
# 1. FILTRAGE PAR VARIANCE (Traitement par Colonnes via Indexation)
# ==============================================================================
print("1. Filtrage par Variance sur les features par blocs (Anti-RAM)...")

selector = VarianceThreshold(threshold=0.01)
selected_feature_indices = []

# La clé est l'indexation [:, chunk_start:chunk_end] directement sur l'objet np.load
for i in tqdm(range(0, n_features, CHUNK_SIZE), desc="Filtrage par Variance"):
    chunk_start = i
    chunk_end = min(i + CHUNK_SIZE, n_features)
    
    # Charger SEULEMENT ce bloc de features de X_train
    current_chunk = data['X_train'][:, chunk_start:chunk_end]
    
    selector.fit(current_chunk)
    
    local_indices = selector.get_support(indices=True)
    global_indices = [idx + chunk_start for idx in local_indices]
    selected_feature_indices.extend(global_indices)
    
    del current_chunk 

print(f"Features conservées après le filtre de variance : {len(selected_feature_indices):,}")
print("-" * 50)


# ==============================================================================
# 2. CONSTRUCTION DU JEU DE DONNÉES RÉDUIT ET STANDARDISATION
# ==============================================================================
print("2. Construction et Standardisation du jeu de données réduit...")

# Chargement de la matrice finale réduite (doit être gérable en RAM maintenant)
X_train_reduced = data['X_train'][:, selected_feature_indices]
X_test_reduced = data_test['X_test'][:, selected_feature_indices]

print(f"Dimensions réduites finales : {X_train_reduced.shape}")

# Fermer les fichiers npz
data.close()
data_test.close()

# Standardisation OBLIGATOIRE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)


# ==============================================================================
# 3. GRID SEARCH ELASTIC NET (Sélection Fine)
# ==============================================================================
print("3. Grid Search pour l'Elastic Net (Sélection des coefficients)....")

param_grid = {
    'alpha': [1e-4, 1e-3, 1e-2, 0.1],
    'l1_ratio': [0.1, 0.5, 0.9, 1.0]
}

elastic_net_base = SGDClassifier(
    loss='log_loss', penalty='elasticnet', random_state=42, max_iter=1000, tol=1e-3
)

# ... (Dans la section 3: Grid Search ELASTIC NET)
grid_search = GridSearchCV(
    estimator=elastic_net_base,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=0,
    # === CHANGEMENT CRITIQUE ANTI-RAM ===
    n_jobs=1  # Désactive la parallélisation pour éviter les copies multiples
    # ===================================
)

grid_search.fit(X_train_scaled, y_train)

best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']

print("--------------------------------------------------")
print("⭐ Meilleurs hyperparamètres Elastic Net trouvés :")
print(f"  Alpha (Force) : {best_alpha}")
print(f"  L1 Ratio (Mélange L1/L2) : {best_l1_ratio}")
print(f"  Meilleur score F1 : {grid_search.best_score_:.4f}")
print("--------------------------------------------------")
print("-" * 50)


# ==============================================================================
# 4. SÉLECTION FINALE DES 5000 FEATURES (Elastic Net)
# ==============================================================================
print(f"4. Sélection finale des {final_k:,} features...")

final_selector_model = SGDClassifier(
    loss='log_loss', penalty='elasticnet', alpha=best_alpha, l1_ratio=best_l1_ratio,
    random_state=42, max_iter=1000, tol=1e-3
)
final_selector_model.fit(X_train_scaled, y_train)

coefs = np.abs(final_selector_model.coef_[0])
top_indices_final = np.argsort(coefs)[::-1][:final_k]

X_train_final = X_train_scaled[:, top_indices_final]
X_test_final = X_test_scaled[:, top_indices_final]

print(f"Dimensions finales de X_train : {X_train_final.shape}")
print("-" * 50)


# ==============================================================================
# 5. ENTRAÎNEMENT XGBOOST AVEC SUPPORT GPU
# ==============================================================================
print("5. Entraînement du modèle XGBoost (GPU activé)...")

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

print("Modèle XGBoost entraîné.")
print("-" * 50)


# ==============================================================================
# 6. PRÉDICTION ET FICHIER DE SOUMISSION
# ==============================================================================
y_pred = xgb.predict(X_test_final)

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = "submission_final_grev_path.csv"
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")