import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
import os

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION ET CHARGEMENT DES FICHIERS INPUT (Chemins GREV)
# ==============================================================================

# VÉRIFIEZ ET AJUSTEZ CE CHEMIN vers votre dossier contenant train.npz et test.npz
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

# Chargement des petits vecteurs
y_train = data['y_train']
ids_test = data_test['ids']
n_samples = y_train.shape[0]
n_features = data['X_train'].shape[1]

final_k = 5000
CHUNK_SIZE = 100000 

print(f"Dimensions cibles : ({n_samples}, {n_features:,})")
print("-" * 50)


# ==============================================================================
# 1. CALCUL DE LA VARIANCE ET SÉLECTION DU TOP 5000 (Anti-RAM)
# ==============================================================================
print(f"1. Calcul de la Variance pour la sélection du TOP {final_k:,} features...")

all_feature_variances = []

# Itération sur les 1,000,000 features par blocs de 100,000
for i in tqdm(range(0, n_features, CHUNK_SIZE), desc="Calcul de la Variance par blocs"):
    chunk_start = i
    chunk_end = min(i + CHUNK_SIZE, n_features)
    
    # 💥 Le mode Anti-RAM : on ne charge que le bloc de 100k colonnes en RAM
    current_chunk = data['X_train'][:, chunk_start:chunk_end]
    
    # Calcul de la variance pour ce bloc de 100k features
    current_variances = np.var(current_chunk, axis=0) 
    all_feature_variances.append(current_variances)
    
    # Libération explicite du bloc de features
    del current_chunk 

# Combinaison de toutes les variances en un seul vecteur
combined_variances = np.concatenate(all_feature_variances)

# Sélection des indices des 5000 plus grandes variances (la clé du tri)
# np.argsort trie par ordre croissant; [::-1] inverse pour obtenir l'ordre décroissant.
top_k_indices = np.argsort(combined_variances)[::-1][:final_k]

print(f"Features sélectionnées : {len(top_k_indices):,}")
print("-" * 50)


# ==============================================================================
# 2. CONSTRUCTION DU JEU DE DONNÉES FINAL ET STANDARDISATION
# ==============================================================================
print("2. Construction du jeu de données final et Standardisation...")

# On charge maintenant uniquement les 5000 colonnes choisies (taille gérable en RAM)
X_train_final = data['X_train'][:, top_k_indices]
X_test_final = data_test['X_test'][:, top_k_indices]

print(f"Dimensions finales (X_train) : {X_train_final.shape}")

# Fermer les fichiers npz
data.close()
data_test.close()

# Standardisation OBLIGATOIRE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

del X_train_final, X_test_final # Libérer la mémoire des matrices non normalisées
print("-" * 50)


# ==============================================================================
# 3. ENTRAÎNEMENT XGBOOST AVEC SUPPORT GPU
# ==============================================================================
print("3. Entraînement du modèle XGBoost...")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=3,
    reg_lambda=1.0,
    tree_method='hist',  # Utilise le GPU si disponible (sinon CPU)
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

# L'entraînement devrait être beaucoup plus rapide sur 5000 features
xgb.fit(X_train_scaled, y_train)

print("Modèle XGBoost entraîné sur le TOP 5000 features par variance.")
print("-" * 50)


# ==============================================================================
# 4. PRÉDICTION ET FICHIER DE SOUMISSION
# ==============================================================================
y_pred = xgb.predict(X_test_scaled)

submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = "submission_final_top_5000_variance.csv"
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")