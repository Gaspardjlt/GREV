import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier # Utilisé pour la sélection L2
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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

# final_k n'est plus fixe, il sera déterminé par la grid search
CHUNK_SIZE = 100000

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
    # data['X_train'] est un mmap, on le lit par blocs
    current_chunk = data['X_train'][:, chunk_start:chunk_end]
    current_variances = np.var(current_chunk, axis=0)
    all_feature_variances.append(current_variances)
    del current_chunk

combined_variances = np.concatenate(all_feature_variances)

# On garde toutes les features au-dessus du seuil 0.02
safe_indices = np.where(combined_variances > 0.02)[0]
# Chargement des données réduites en RAM
X_train_reduced = data['X_train'][:, safe_indices]
X_test_reduced = data_test['X_test'][:, safe_indices]

print(f"Features conservées après le pré-filtrage (taille gérable) : {X_train_reduced.shape[1]:,}")
print("-" * 50)


# ==============================================================================
# 2. STANDARDISATION ET ENTRAÎNEMENT RIDGE (L2)
# ==============================================================================
print("2. Standardisation, Split de validation et Entraînement du Modèle L2 (Ridge)...")

# Standardisation de la matrice réduite (création de X_train_scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Séparation des données d'entraînement pour la validation interne (80% train, 20% validation)
print("Séparation des données d'entraînement pour la validation (80/20)...")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Entraînement du Modèle Ridge pour obtenir les coefficients de scoring
ridge_model = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.01,
    random_state=42, max_iter=1000, tol=1e-3,
    n_jobs=1
)

ridge_model.fit(X_train_scaled, y_train)

# Récupération et tri des indices par importance (valeur absolue des coefficients)
top_indices_ridge = np.argsort(np.abs(ridge_model.coef_[0]))[::-1]
print("Modèle Ridge entraîné. Coefficients prêts pour le classement.")
print("-" * 50)


# ==============================================================================
# 3. SÉLECTION ET OPTIMISATION DU NOMBRE DE FEATURES (GRID SEARCH)
# ==============================================================================
print("3. Optimisation du nombre de features (Grid Search L2-XGBoost)...")

# Range spécifié: 1500 à 6500 par pas de 500
k_values = range(1500, 6501, 500)
best_k = 0
best_auc = -1.0
auc_scores = {}

# Réglage de XGBoost pour la recherche rapide (moins d'estimateurs)
xgb_val_params = {
    'n_estimators': 150, # Réduit pour accélérer la Grid Search
    'max_depth': 6,
    'learning_rate': 0.1,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0
}

for k in tqdm(k_values, desc="Grid Search sur k features"):
    # 3.1. Sélection des top k features pour cette itération
    current_top_indices = top_indices_ridge[:k]

    # 3.2. Création des matrices d'entraînement et de validation avec k features
    # Utilisation des données splittées plus tôt (X_tr et X_val)
    X_tr_k = X_tr[:, current_top_indices]
    X_val_k = X_val[:, current_top_indices]

    # 3.3. Entraînement XGBoost
    xgb_val = XGBClassifier(**xgb_val_params)
    xgb_val.fit(X_tr_k, y_tr)

    # 3.4. Évaluation sur l'ensemble de validation (AUC)
    y_pred_proba_val = xgb_val.predict_proba(X_val_k)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba_val)
    auc_scores[k] = auc_score

    # 3.5. Mise à jour du meilleur k
    if auc_score > best_auc:
        best_auc = auc_score
        best_k = k

    # Affiche le résultat pour la console (dans la barre de tqdm ou après)
    # Le print est déplacé après la boucle pour un affichage plus propre
    pass

print("\n--- Résultats de la Grid Search ---")
for k, auc in auc_scores.items():
    print(f"  -> k={k:,}: AUC = {auc:.6f} {'(MEILLEUR)' if k == best_k else ''}")

print(f"\n✅ MEILLEUR RÉSULTAT OPTIMISÉ: {best_k:,} features avec AUC de {best_auc:.6f}")
print("-" * 50)


# ==============================================================================
# 4. ENTRAÎNEMENT FINAL XGBOOST AVEC LE MEILLEUR K
# ==============================================================================

# Utiliser le k optimal pour l'entraînement final
final_k = best_k

# 4.1 Sélection finale avec le meilleur K sur les ensembles complets
final_top_indices = top_indices_ridge[:final_k]
X_train_final = X_train_scaled[:, final_top_indices]
X_test_final = X_test_scaled[:, final_top_indices]

# Libération de la mémoire des matrices intermédiaires
del X_train_reduced, X_test_reduced, X_train_scaled, X_test_scaled, X_tr, X_val, y_tr, y_val

# data.close() et data_test.close() sont placés ici après que toutes les variables en mémoire aient été créées
data.close()
data_test.close()

print(f"4. Entraînement final du modèle XGBoost avec les {final_k:,} features sélectionnées...")
print(f"Dimensions finales (X_train) : {X_train_final.shape}")

# Entraînement final (on revient à 300 estimateurs pour le modèle de soumission)
xgb = XGBClassifier(
    n_estimators=300, # Nombre original d'estimateurs
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

print(f"Modèle XGBoost entraîné sur le TOP {final_k:,} features par score L2.")
print("-" * 50)


# ==============================================================================
# 5. PRÉDICTION ET FICHIER DE SOUMISSION
# ==============================================================================
y_pred = xgb.predict(X_test_final)

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = f"submission_final_top_{final_k:,}_L2_optimized.csv".replace(",", "")
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")