import numpy as np

# Ouvre le fichier .npz
data = np.load("inf-8245-fall-2025/train.npz")

X_train = data['X_train']
y_train = data['y_train']
ids = data['ids']

import matplotlib.pyplot as plt

variabilite = X_train.std()
variabilite.plot(kind='bar', figsize=(10,5))
plt.title("Variabilité (écart-type) de chaque variable dans X_train")
plt.ylabel("Écart-type")
plt.xlabel("Variables")
plt.show()


# print("X_train:", X_train)
# print("y_train:", y_train)
# print("ids:", ids)
