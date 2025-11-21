import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd

data = np.load("inf-8245-fall-2025/train.npz")

X_train = data['X_train']
y_train = data['y_train']
ids = data['ids']

# Étape 1: Identifier les valeurs négatives dans X_train
# np.any(X_train < 0, axis=0) retourne un tableau booléen.
# Chaque élément est True si la colonne correspondante contient AU MOINS une valeur négative.
negative_check_per_column = np.any(X_train < 0, axis=0)

# Étape 2: Compter le nombre de colonnes avec au moins une valeur négative
# np.sum() sur un tableau booléen compte le nombre de valeurs True.
nombre_colonnes_negatives = np.sum(negative_check_per_column)

print(f"Nombre de colonnes avec au moins une valeur négative: {nombre_colonnes_negatives}")