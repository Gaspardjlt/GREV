import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
from sklearn.model_selection import train_test_split

# def regroup_top_n(df, column_name, n=5):
#     """
#     Regroupe toutes les modalités d'une colonne sauf les 'n' plus fréquentes
#     dans une nouvelle catégorie 'Others'.
#     """
#     # 1. Calculer les 5 modalités les plus fréquentes (récurrentes)
#     top_n_values = df[column_name].value_counts().nlargest(n).index

#     # 2. Créer la nouvelle colonne modifiée
#     # Si la valeur est dans le top 5, on garde la valeur. Sinon, on met 'Others'.
#     df[f'{column_name}_Top'] = df[column_name].apply(
#         lambda x: x if x in top_n_values else 'Others'
#     )
#     return df

def regroup_top_n(df, column_name, n=5):
    """
    Regroupe toutes les modalités d'une colonne sauf les 'n' plus fréquentes
    dans une nouvelle catégorie 'Others', en utilisant .loc pour éviter le SettingWithCopyWarning.
    """
    # ... le reste du code est inchangé
    top_n_values = df[column_name].value_counts().nlargest(n).index

    print(top_n_values)

    # 2. Générer la SÉRIE de valeurs transformées en utilisant .apply()
    transformed_series = df[column_name].apply(
        lambda x: x if x in top_n_values else f'{column_name}_Others'
    )
    
    # 3. Utiliser .loc pour affecter la SÉRIE transformée à la nouvelle colonne
    # C'est la syntaxe standard pour créer ou modifier une colonne avec .loc et une série.
    df.loc[:, f'{column_name}_Top'] = transformed_series

    df = df.drop(columns=[column_name])
    return df

cols_to_keep = [
    'Location',
    'Isolation source',
    'Creation Year', # La variable dérivée
]

# Appliquer le regroupement pour chaque colonne

metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
data = np.load("inf-8245-fall-2025/train.npz")
y = data['y_train'] 

# Conversion et création de la variable temporelle catégorielle (facultatif mais recommandé)
metadata['Create date'] = pd.to_datetime(metadata['Create date'], errors='coerce')
metadata['Creation Year'] = metadata['Create date'].dt.year

meta = metadata[cols_to_keep]

# meta = pd.get_dummies(meta, columns=[
#     'Location',
#     'Isolation source',
#     'Creation Year', # La variable dérivée
# ],drop_first=True)

for col in [
    'Location',
    'Isolation source'
]:
    meta = regroup_top_n(meta, col, n=5)

top = [
    'Location_Top',
    'Isolation source_Top',
    'Creation Year'
]
meta = meta[top]

top_columns_to_encode = [
    'Location_Top',
    'Isolation source_Top',
]

# Application du One-Hot Encoding
meta_encoded = pd.get_dummies(
    meta, 
    columns=top_columns_to_encode, 
    drop_first=True  # Pour éviter la multicolinéarité
)

print(meta)

X_train, X_test, y_train, y_test = train_test_split(
    meta, y, test_size=0.4, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_hat = predictions = model.predict(X_test)

# 3. Calculer le F1-Score
f1 = f1_score(y_test, y_hat) 
print(f"Le F1-Score (pour la classe positive) est: {f1:.4f}")

# 4. Afficher le rapport complet (recommandé)
# Le classification_report affiche le F1-Score pour chaque classe, ainsi que la Précision et le Rappel
print("\n--- Rapport de Classification ---")
print(classification_report(y_test, y_hat))



