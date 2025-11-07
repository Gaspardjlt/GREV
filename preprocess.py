import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

data = np.load("GREV/inf-8245-fall-2025/train.npz")

X_train = data['X_train']
y_train = data['y_train']
ids = data['ids']


variabilite = X_train.std(axis=0)
# Nombre de colonnes constantes
nb_constantes = np.sum(variabilite == 0)
print("Nombre de colonnes constantes :", nb_constantes)
# (Optionnel) Indices des colonnes constantes
indices_constantes = np.where(variabilite == 0)[0]
X_train = np.delete(X_train, np.where(variabilite == 0)[0], axis=1)

