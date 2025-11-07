# import numpy as np
# import matplotlib.pyplot as plt

# data = np.load("../inf-8245-fall-2025/train.npz")

# X_train = data["X_train"].astype(np.int8)  # si c’est binaire
# y_train = data["y_train"]
# ids = data["ids"]

# chunk_size = 10000
# variabilite_parts = []

# n_features = X_train.shape[1]
# print(f"Calcul de la variabilité sur {n_features} features en paquets de {chunk_size}...")

# for i in range(0, n_features, chunk_size):
#     chunk = X_train[:, i:i + chunk_size]
#     # Calcul de la std colonne par colonne (axis=0)
#     chunk_std = np.std(chunk, axis=0)
#     variabilite_parts.append(chunk_std)
#     print(f" → Chunk {i // chunk_size + 1} traité ({min(i + chunk_size, n_features)} colonnes max)")

# # On concatène toutes les std en un seul vecteur
# variabilite = np.concatenate(variabilite_parts)

# # On récupère les indices triés par ordre décroissant
# sorted_indices = np.argsort(-variabilite)

# # On affiche les 10 plus variables
# top_indices = sorted_indices[:100000]
# # print("\nIndices des 10 features les plus variables :", top_indices)
# # print("Valeurs correspondantes :", variabilite[top_indices])

# # Visualisation
# # plt.figure(figsize=(10,5))
# # plt.bar(range(10), variabilite[top_indices], color='orange')
# # plt.title("Top 10 des features avec la plus forte variabilité")
# # plt.ylabel("Écart-type")
# # plt.xlabel("Index de features")
# # plt.show()

# data.close()

# import numpy as np
# from sklearn.feature_selection import f_classif
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# # === Chargement du .npz ===
# data = np.load("../inf-8245-fall-2025/train.npz")
# X_train = data["X_train"]
# y_train = data["y_train"]

# chunk_size = 10000
# n_features = X_train.shape[1]

# f_scores = np.zeros(n_features)
# p_values = np.ones(n_features)

# print(f"Calcul des scores F (ANOVA) sur {n_features} features en paquets de {chunk_size}...")

# # === Boucle sur les features par blocs ===
# for i in tqdm(range(0, n_features, chunk_size)):
#     chunk = X_train[:, i:i + chunk_size]

#     # On convertit seulement ce bloc en float32 (économie de RAM)
#     chunk = chunk.astype(np.float32)

#     scores, pvals = f_classif(chunk, y_train)
#     f_scores[i:i + chunk.shape[1]] = scores
#     p_values[i:i + chunk.shape[1]] = pvals

# # === Sélection des meilleures features ===
# sorted_indices = np.argsort(-f_scores)  # tri décroissant
# k = 1000
# top_indices = sorted_indices[:k]

# print(f"\nTop {k} features les plus discriminantes (ANOVA F-test) :")
# print(top_indices)

# # === Visualisation des 20 meilleures ===
# plt.figure(figsize=(10,5))
# plt.bar(range(20), f_scores[top_indices[:20]], color='orange')
# plt.title("Top 20 features avec le score F le plus élevé (ANOVA)")
# plt.ylabel("Score F")
# plt.xlabel("Index de feature")
# plt.show()

# data.close()

import numpy as np
from sklearn.feature_selection import f_classif
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Chargement du .npz ===
data = np.load("../inf-8245-fall-2025/train.npz")
data_test = np.load("../inf-8245-fall-2025/test.npz")
X_train = data['X_train']
X_test = data_test['X_test']
y_train = data['y_train']
ids = data['ids']

chunk_size = 10000
n_features = X_train.shape[1]

f_scores = np.zeros(n_features)
p_values = np.ones(n_features)

print(f"Calcul des scores F (ANOVA) sur {n_features} features en paquets de {chunk_size}...")

# === Boucle sur les features par blocs ===
for i in tqdm(range(0, n_features, chunk_size)):
    chunk = X_train[:, i:i + chunk_size]

    # Conversion en float32 (gain mémoire)
    chunk = chunk.astype(np.float32)

    # Calcul du F-score pour ce bloc
    scores, pvals = f_classif(chunk, y_train)

    # On stocke les résultats dans les bons emplacements
    f_scores[i:i + chunk.shape[1]] = scores
    p_values[i:i + chunk.shape[1]] = pvals

# === Sélection des meilleures features ===
k = 10000  # nombre de features à garder
sorted_indices = np.argsort(-f_scores)  # tri décroissant
top_indices = sorted_indices[:k]

print(f"\nSélection des {k:,} features les plus discriminantes...")

# === Réduction du jeu de données ===
X_train_reduced = X_train[:, top_indices]

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ID | Label  
submission = pd.DataFrame({
    "id": data_test['ids'],   
    "label": y_pred.astype(int) 
})

submission.to_csv("submission.csv", index=False)

print("✅ Fichier 'submission.csv' prêt à être soumis sur Kaggle !")

data.close()


