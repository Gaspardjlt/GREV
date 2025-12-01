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
data = np.load("inf-8245-fall-2025/train.npz")
X_train = data["X_train"]
y_train = data["y_train"]

data_test = np.load("inf-8245-fall-2025/test.npz")
X_test = data_test["X_test"]
ids_test = data_test['ids']

n_samples = y_train.shape[0]
n_features = data['X_train'].shape[1]

final_k = 5000
CHUNK_SIZE = 100000 

print(f"Dimensions cibles : ({n_samples}, {n_features:,})")
print("-" * 50)


# Standardisation OBLIGATOIRE
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

print("1. Calcul des statistiques (Moyenne/Écart-type) par blocs...")
# VECTEURS D'ACCUMULATION
sum_X = np.zeros(n_features)
sum_sq_X = np.zeros(n_features)
n_samples = X_train.shape[0]

# Chargement par blocs pour calculer la SOMME et la SOMME DES CARRÉS
for i in tqdm(range(0, n_features, CHUNK_SIZE), desc="Calcul des sommes"):
    chunk_start = i
    chunk_end = min(i + CHUNK_SIZE, n_features)
    
    # CHARGEMENT DU BLOC DE DONNÉES BRUTES
    current_chunk = data['X_train'][:, chunk_start:chunk_end]
    
    # Accumulation des statistiques
    sum_X[chunk_start:chunk_end] = np.sum(current_chunk, axis=0)
    sum_sq_X[chunk_start:chunk_end] = np.sum(current_chunk**2, axis=0)
    
    del current_chunk

# Calcul de la moyenne et de l'écart-type globaux (1M de valeurs)
global_mean = sum_X / n_samples
global_std = np.sqrt((sum_sq_X / n_samples) - global_mean**2)
# Éviter la division par zéro dans la std
global_std[global_std == 0] = 1.0 # ou un petit epsilon

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
    
    # Standardisation IN-PLACE (dans le chunk)
    mu = global_mean[chunk_start:chunk_end]
    sigma = global_std[chunk_start:chunk_end]
    
    # Standardisation Z-score
    current_chunk_scaled = (current_chunk - mu) / sigma
    
    # Calcul de la variance du bloc standardisé
    current_variances = np.var(current_chunk_scaled, axis=0) 
    all_feature_variances.append(current_variances)
    
    del current_chunk, current_chunk_scaled

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
X_train_final_brut = data['X_train'][:, top_k_indices]
X_test_final_brut = data_test['X_test'][:, top_k_indices]

print(f"Dimensions finales (X_train) : {X_train_final_brut.shape}")

# Standardisation des 5000 colonnes sélectionnées en utilisant les statistiques globales
mu_final = global_mean[top_k_indices]
sigma_final = global_std[top_k_indices]

# Transformation Z-score finale
# **C'est là qu'on applique la standardisation pour le modèle**
X_train_final_scaled = (X_train_final_brut - mu_final) / sigma_final
X_test_final_scaled = (X_test_final_brut - mu_final) / sigma_final


print(f"Dimensions finales (X_train) : {X_train_final_scaled.shape}")

# Fermer les fichiers npz
data.close()
data_test.close()



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ... (Après la sélection des top_k_indices) ...

# 2. Construction du jeu de données FINAL standardisé
# On charge les colonnes choisies (top_k_indices) à partir des matrices SCALED
# Pour l'Anti-RAM, vous devrez peut-être recharger X_train et X_test puis
# les standardiser pour les 5000 colonnes sélectionnées, mais le plus simple est :

# 3. Entraînement de la QDA
print("3. Entraînement de la QDA...")

# ATTENTION: Si 5000 est encore trop grand pour la QDA, vous devrez utiliser
# une version régularisée de la QDA (shrinkage) ou réduire le k.
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_final_scaled, y_train)
y_pred = qda.predict(X_test_final_scaled)

print("QDA entraînée avec succès.")


submission = pd.DataFrame({
    "id": ids_test,
    "label": y_pred.astype(int)
})

submission_filename = "submission_final_top_5000_variance_28_11.csv"
submission.to_csv(submission_filename, index=False)

print(f"✅ Fichier de soumission créé : {submission_filename}")

