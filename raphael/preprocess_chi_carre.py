import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# --- 1. Chargement des Données ---
print("1. Chargement des données...")
# Assurez-vous que le chemin d'accès au fichier est correct.
# J'utilise le chemin fourni dans votre contexte initial :
data = np.load("inf-8245-fall-2025/train.npz")

X_train = data['X_train']
y_train = data['y_train']
ids = data['ids']

# Afficher les dimensions pour vérification
print(f"Dimensions de X_train : {X_train.shape}")
print(f"Dimensions de y_train : {y_train.shape}")
print("-" * 30)

# --- 2. Vérification Préalable (Non-Négativité) ---
# Confirmation que X_train ne contient que des valeurs non-négatives (condition pour chi2)
nombre_colonnes_negatives = np.sum(np.any(X_train < 0, axis=0))

print("2. Vérification de la condition Chi-Carré...")
if nombre_colonnes_negatives == 0:
    print("✅ Condition remplie : Aucune colonne ne contient de valeur négative.")
else:
    # Ceci est une mesure de sécurité, même si vous avez confirmé le résultat 0.
    print(f"⚠️ Attention : {nombre_colonnes_negatives} colonnes contiennent des valeurs négatives.")
    print("Le Chi-Carré nécessite des données non-négatives. Un prétraitement pourrait être nécessaire.")
print("-" * 30)


# --- 3. Sélection de Features avec le Chi-Carré (SelectKBest) ---
# Déterminer le nombre de features à garder
# C'est un hyperparamètre, 50 est un point de départ raisonnable.
k_features_to_select = 5000

print(f"3. Application du Chi-Carré pour sélectionner {k_features_to_select} features...")

# Initialiser le sélecteur :
# score_func=chi2 spécifie la fonction de scoring (Chi-Carré)
# k=k_features_to_select spécifie le nombre de meilleures features à garder
selector = SelectKBest(score_func=chi2, k=k_features_to_select)

# Ajuster le sélecteur et transformer les données
# L'ajustement calcule le score Chi-Carré pour chaque feature
X_train_selected = selector.fit_transform(X_train, y_train)

# --- 4. Affichage des Résultats ---

# Les scores Chi-Carré pour toutes les features
chi2_scores = selector.scores_
# Les p-values (plus la p-value est faible, plus la relation est significative)
p_values = selector.pvalues_

print(f"Features originales : {X_train.shape[1]}")
print(f"Features sélectionnées : {X_train_selected.shape[1]}")

# Pour obtenir les indices des features sélectionnées (utile pour l'interprétation)
selected_indices = selector.get_support(indices=True)
print(f"\nIndices des {k_features_to_select} features les plus pertinentes :")
# Afficher les 10 premiers indices et les 10 derniers
print(f"Début : {selected_indices[:10]}")
print(f"Fin : {selected_indices[-10:]}")
print("-" * 30)

# Vous pouvez également afficher les meilleurs scores et p-values
top_k_indices = np.argsort(chi2_scores)[::-1][:k_features_to_select]
print("Top 5 Scores Chi-Carré et P-Values :")
for i in top_k_indices[:5]:
    print(f"Feature {i}: Score Chi2 = {chi2_scores[i]:.2f}, P-Value = {p_values[i]:.4e}")

print("\nOpération de sélection de features terminée. X_train_selected est prêt pour l'entraînement.")