import numpy as np
import os

print(os.getcwd())
# Ouvre le fichier .npz
data = np.load("GREV/inf-8245-fall-2025/train.npz")

X_train = data['X_train']
X_test = data_test['X_test']
y_train = data['y_train']
ids = data['ids']

X_train  = X_train[:, :10000]  # prend seulement 10 000 colonnes
X_test = X_test[:, :10000]

import matplotlib.pyplot as plt
import pandas as pd


variabilite = X_train.std(axis=0)

# Tri décroissant
indices_trie = np.argsort(variabilite)[::-1]  # du plus variable au moins variable

# Sélection des 100 moins variables (fin du tableau)
indices_moins_variables = indices_trie[-100:]

# Trie ces 100 variables dans l’ordre décroissant (par lisibilité)
indices_moins_variables = indices_moins_variables[np.argsort(variabilite[indices_moins_variables])[::-1]]

# Récupère les valeurs correspondantes
variabilite_moins_variables = variabilite[indices_moins_variables]

# Tri décroissant
indices_trie = np.argsort(variabilite)[::-1]  # du plus variable au moins variable

# Sélection des 100 moins variables (fin du tableau)
indices_moins_variables = indices_trie[-100:]

# Trie ces 100 variables dans l’ordre décroissant (par lisibilité)
indices_moins_variables = indices_moins_variables[np.argsort(variabilite[indices_moins_variables])[::-1]]

# Récupère les valeurs correspondantes
variabilite_moins_variables = variabilite[indices_moins_variables]

print(variabilite_moins_variables)
print(len())


# pd.Series(variabilite).plot(kind='bar', figsize=(10, 5))
# plt.title("Variabilité (écart-type) de chaque variable dans X_train")
# plt.ylabel("Écart-type")
# plt.xlabel("Index de l'échantillon")
# plt.show()



# print("X_train:", X_train)
# print("y_train:", y_train)
# print("ids:", ids)
