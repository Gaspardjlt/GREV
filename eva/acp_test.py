import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

data = np.load("inf-8245-fall-2025/train.npz")

X_train = data['X_train'].astype(np.float32)
y_train = data['y_train']
ids = data['ids']


# variabilite = X_train.std(axis=0)
# # Nombre de colonnes constantes
# nb_constantes = np.sum(variabilite == 0)
# print("Nombre de colonnes constantes :", nb_constantes)
# # (Optionnel) Indices des colonnes constantes
# indices_constantes = np.where(variabilite == 0)[0]
# X_train = np.delete(X_train, np.where(variabilite == 0)[0], axis=1)

pca = PCA(n_components=250, svd_solver="randomized")
X_red = pca.fit_transform(X_train)

# Sauvegarde
chemin = r"C:/Users/Eva/Desktop/X_reduit.npz"

# Ou sous Linux/Mac : chemin = "/home/eva/Desktop/X_reduit.npz"

# Sauvegarde
np.savez_compressed(chemin, X=X_red)