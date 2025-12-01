# Exemple complet : comparaison de modèles avec préprocessing adapté
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


# Charger les données
data = np.load("inf-8245-fall-2025/train.npz")
X = data['X_train']
y = data['y_train']

# Calcul du taux de sparsité
zero_values = (X == 0).sum()
total_values = X.shape[0] * X.shape[1]
sparsity = zero_values / total_values

print(f"Taux de sparsité : {sparsity*100:.2f}%")


# X  = X[:, :10000] 

# # 2️⃣ Identifier les colonnes numériques et catégorielles
# # Ici toutes les features sont numériques, mais en général :
# numeric_features = np.arange(X.shape[1])
# categorical_features = []

# # 3️⃣ Préprocessors
# # Pour modèles sensibles à l'échelle
# scaler = StandardScaler()
# preprocessor_scaled = ColumnTransformer(
#     transformers=[
#         ('num', scaler, numeric_features)
#     ]
# )

# # Pour arbres (pas de scaling)
# preprocessor_none = ColumnTransformer(
#     transformers=[
#         ('num', 'passthrough', numeric_features)
#     ]
# )

# # 4️⃣ Définir les modèles avec pipelines adaptés
# models = {
#     "RandomForest": Pipeline([
#         ('preproc', preprocessor_none),
#         ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
#     ]),
#     "Bagging": Pipeline([
#         ('preproc', preprocessor_none),
#         ('clf', BaggingClassifier(n_estimators=100, random_state=42))
#     ]),
#     "GradientBoosting": Pipeline([
#         ('preproc', preprocessor_none),
#         ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
#     ]),
#     "LogisticRegression": Pipeline([
#         ('preproc', preprocessor_scaled),
#         ('clf', LogisticRegression(max_iter=1000, random_state=42))
#     ]),
#     "SVM": Pipeline([
#         ('preproc', preprocessor_scaled),
#         ('clf', SVC(kernel='rbf', gamma='scale', random_state=42))
#     ]),
#     "LDA": Pipeline([
#         ('preproc', preprocessor_scaled),
#         ('clf', LinearDiscriminantAnalysis())
#     ]),
#     "QDA": Pipeline([
#         ('preproc', preprocessor_scaled),
#         ('clf', QuadraticDiscriminantAnalysis())
#     ]),
#     "NeuralNetwork": Pipeline([
#         ('preproc', preprocessor_scaled),
#         ('clf', MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, random_state=42))
#     ])
# }

# # 5️⃣ Séparer train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 6️⃣ Évaluer chaque modèle avec cross-validation et test set
# from sklearn.metrics import accuracy_score

# results = {}
# for name, pipeline in models.items():
#     # Cross-validation
#     cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
#     # Entraînement et score sur test
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     test_acc = accuracy_score(y_test, y_pred)
    
#     results[name] = {
#         "CV Mean Accuracy": np.mean(cv_scores),
#         "Test Accuracy": test_acc
#     }

# # 7️⃣ Afficher les résultats
# import pandas as pd
# df_results = pd.DataFrame(results).T
# print(df_results.sort_values("Test Accuracy", ascending=False))

