import numpy as np

# Ouvre le fichier .npz
data = np.load("inf-8245-fall-2025/train.npz")
data_test = np.load("inf-8245-fall-2025/test.npz")

print(data_test)

X_train = data['X_train']
X_test = data_test['X_test']
y_train = data['y_train']
ids = data['ids']

X_train  = X_train[:, :10000]  # prend seulement 10 000 colonnes
X_test = X_test[:, :10000]

import matplotlib.pyplot as plt

# X_train = X_train.astype('float32')
# variabilite = X_train.std()
# variabilite.plot(kind='bar', figsize=(10,5))
# plt.title("Variabilité (écart-type) de chaque variable dans X_train")
# plt.ylabel("Écart-type")
# plt.xlabel("Variables")
# plt.show()
import numpy as np

# # Indices des colonnes constantes
# constantes = np.where(np.apply_along_axis(lambda col: np.all(col == col[0]), 0, X_train))[0]

# print("Nombre de colonnes constantes :", len(constantes))
# print("Indices des colonnes constantes :", constantes)

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


# print("X_train:", X_train)
# print("y_train:", y_train)
# print("ids:", ids)
