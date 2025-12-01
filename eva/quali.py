import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency

metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
data = np.load("inf-8245-fall-2025/train.npz")
y = data['y_train']
metadata['Y'] = y

import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize=(10, 6))
# # Compter la fréquence de chaque catégorie dans 'Organism group'
# sns.countplot(data=metadata, y='Organism group', order=metadata['Organism group'].value_counts().index)
# plt.title('Distribution de la variable Organism group')
# plt.xlabel('Nombre d\'observations')
# plt.ylabel('Organism group')
# plt.show()

# Croisement de 'Isolation source' et 'Y'

# column_name = 'Isolation type'

# # 1. Nombre total de lignes
# total_rows = len(metadata)

# # 2. Nombre de valeurs manquantes (NaN)
# missing_values = metadata[column_name].isnull().sum()

# # 3. Calcul du pourcentage
# missing_percentage = (missing_values / total_rows) * 100

# print(f"--- Analyse des données manquantes pour la colonne '{column_name}' ---")
# print(f"Nombre total d'observations : {total_rows}")
# print(f"Nombre de valeurs manquantes (NaN) : {missing_values}")
# print(f"Taux de données manquantes : {missing_percentage:.2f}%")

# print("\n--- Aperçu des 5 premières lignes manquantes ---")

# # Affichage des colonnes clés pour inspecter ces lignes
# cols_to_display = ['ID', 'Organism group', column_name, 'Isolation source', 'Y']
# pd.set_option('display.max_columns', None) 
# # Crée un masque booléen qui est VRAI là où la valeur est manquante (NaN)
# missing_isolation_type_mask = metadata[column_name].isnull()

# # Filtre le DataFrame pour n'afficher que ces lignes
# individuals_missing_isolation = metadata[missing_isolation_type_mask]
# print(individuals_missing_isolation.head())

plt.figure(figsize=(12, 7))

# Création de l'histogramme (countplot) avec 'hue'
sns.histplot(
    data=metadata, 
    x='Location',  # Variable catégorielle sur l'axe X
    hue='Y',               # Variable cible pour la couleur (hue)
    multiple='stack',      # Empile les barres pour voir la composition
    shrink=0.8,            # Réduit la largeur des barres pour l'esthétique
    palette='viridis',     # Choix de la palette de couleurs
    stat='count'           # Affiche les comptes (quantités)
)

plt.title('Distribution de la Location, colorée par Y')
plt.xlabel('Isolation Source')
plt.ylabel('Nombre d\'Observations (Count)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Y', loc='upper right', labels=metadata['Y'].unique()) # Ajout manuel de la légende pour plus de clarté
plt.tight_layout()
plt.show()

# crosstab_iso = pd.crosstab(metadata['Isolation source'], metadata['Y'], normalize='index') * 100

# plt.figure(figsize=(12, 7))
# # Diagramme à barres empilées montrant la proportion de Y pour chaque Isolation source
# crosstab_iso.plot(kind='bar', stacked=True, colormap='viridis')
# plt.title('Proportion de Y par Isolation  Source')
# plt.xlabel('Isolation Source')
# plt.ylabel('Proportion (%)')
# plt.xticks(rotation=45, ha='right')
# plt.legend(title='Y', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()



# 1. Convertir en format datetime
# metadata['Create date'] = pd.to_datetime(metadata['Create date'])

# # 2. Extraire une nouvelle variable catégorielle (ex: Année)
# metadata['Creation Year'] = metadata['Create date'].dt.year

# # 3. Visualiser l'impact de l'année sur Y (comme dans l'exemple précédent)
# crosstab_year = pd.crosstab(metadata['Creation Year'], metadata['Y'], normalize='index') * 100

# plt.figure(figsize=(10, 6))
# crosstab_year.plot(kind='bar', stacked=True)
# plt.title('Proportion de Y par Année de Création')
# plt.xlabel('Année de Création')
# plt.ylabel('Proportion (%)')
# plt.show()

# --- 1. FONCTION POUR LE V DE CRAMER ---
# def cramers_v(x, y):
#     """Calcule le V de Cramer entre deux variables catégorielles."""
#     # Création de la table de contingence
#     confusion_matrix = pd.crosstab(x, y)
    
#     # Test du Chi-carré
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
    
#     # Correction pour la taille de la matrice (phi max)
#     r, k = confusion_matrix.shape
#     phi_max = min(r - 1, k - 1)
    
#     # Gestion du cas où phi_max = 0 pour éviter la division par zéro
#     if phi_max == 0:
#         return 0.0 
    
#     # Calcul du V de Cramer
#     v = np.sqrt(chi2 / (n * phi_max))
#     return v

# --- 2. PRÉPARATION DES DONNÉES ---

# Assurez-vous d'avoir les données chargées comme suit:
# metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
# data = np.load("inf-8245-fall-2025/train.npz")
# metadata['Y'] = data['y_train'] 

# Conversion et création de la variable temporelle catégorielle (facultatif mais recommandé)
# metadata['Create date'] = pd.to_datetime(metadata['Create date'], errors='coerce')
# metadata['Creation Year'] = metadata['Create date'].dt.year

# # Sélection des colonnes catégorielles pertinentes (+ Y)
# # On exclut 'ID' et la colonne 'Create date' originale
# cols_to_analyze = [
#     'Organism group',
#     'Isolation type',
#     'Location',
#     'Isolation source',
#     'Laboratory typing platform',
#     'Testing standard',
#     'Creation Year', # La variable dérivée
#     'Y' # La variable cible
# ]

# metadata_cat = metadata[cols_to_analyze].copy()

# # S'assurer que toutes les colonnes sont de type catégoriel pour le calcul
# for col in metadata_cat.columns:
#     metadata_cat[col] = metadata_cat[col].astype('category')


# # --- 3. CALCUL DE LA MATRICE DE V DE CRAMER ---

# cols = metadata_cat.columns.tolist()
# n_cols = len(cols)
# cramer_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

# # Remplissage de la matrice
# for i in range(n_cols):
#     for j in range(n_cols):
#         var1 = cols[i]
#         var2 = cols[j]
        
#         if i == j:
#             cramer_matrix.loc[var1, var2] = 1.0 # Diagonale = 1
#         elif i < j:
#             # Calcul et symétrie
#             v_score = cramers_v(metadata_cat[var1], metadata_cat[var2])
#             cramer_matrix.loc[var1, var2] = v_score
#             cramer_matrix.loc[var2, var1] = v_score


# # --- 4. VISUALISATION EN HEATMAP ---

# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     cramer_matrix,
#     annot=True,         # Afficher les valeurs V de Cramer
#     fmt=".2f",          # Formater à 2 décimales
#     cmap="YlGnBu",      # Palette de couleurs
#     linewidths=.5,      # Lignes de séparation
#     cbar_kws={'label': 'V de Cramer (Force de l\'Association)'}
# )
# plt.title('Matrice d\'Association (V de Cramer) entre Variables Catégorielles et Y')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

# # Affichage de la matrice numérique (facultatif)
# print("\nMatrice de V de Cramer:")
# print(cramer_matrix)

# # MEME CHOSE MAIS ON SUPP LES DONNEES MANQUANTES

# # --- 2. ANALYSE DES DONNÉES MANQUANTES ---

# print("--- Analyse des Données Manquantes ---")

# # Nombre total de lignes initiales
# initial_rows = len(metadata_cat)
# print(f"Nombre initial de lignes : {initial_rows}")

# # Calcul du pourcentage de données manquantes par colonne
# missing_percentage = metadata_cat.isnull().sum() / initial_rows * 100
# print("\nPourcentage de données manquantes par variable:")
# print(missing_percentage.sort_values(ascending=False))

# # --- 3. SUPPRESSION DES INDIVIDUS AVEC DONNÉES MANQUANTES (Listwise Deletion) ---

# # Suppression des lignes ayant au moins un NaN
# metadata_cleaned = metadata_cat.dropna(axis=0, how='any')

# # Calcul des lignes enlevées
# final_rows = len(metadata_cleaned)
# rows_removed = initial_rows - final_rows

# print(f"\nNombre de lignes conservées : {final_rows}")
# print(f"Nombre de lignes enlevées : {rows_removed}")
# print(f"Pourcentage de données conservées : {final_rows / initial_rows * 100:.2f}%")


# # --- 4. FONCTION POUR LE V DE CRAMER (réutilisée) ---
# def cramers_v(x, y):
#     """Calcule le V de Cramer entre deux variables catégorielles."""
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     r, k = confusion_matrix.shape
#     phi_max = min(r - 1, k - 1)
    
#     if phi_max == 0:
#         return 0.0 
    
#     v = np.sqrt(chi2 / (n * phi_max))
#     return v

# # --- 5. CALCUL ET VISUALISATION DE LA NOUVELLE MATRICE ---

# print("\n--- Calcul de la Matrice sur les Données Nettoyées ---")

# cols = metadata_cleaned.columns.tolist()
# n_cols = len(cols)
# cramer_matrix_cleaned = pd.DataFrame(index=cols, columns=cols, dtype=float)

# # Remplissage de la matrice avec les données nettoyées
# for i in range(n_cols):
#     for j in range(n_cols):
#         var1 = cols[i]
#         var2 = cols[j]
        
#         if i == j:
#             cramer_matrix_cleaned.loc[var1, var2] = 1.0
#         elif i < j:
#             v_score = cramers_v(metadata_cleaned[var1], metadata_cleaned[var2])
#             cramer_matrix_cleaned.loc[var1, var2] = v_score
#             cramer_matrix_cleaned.loc[var2, var1] = v_score

# # Visualisation
# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     cramer_matrix_cleaned,
#     annot=True,        
#     fmt=".2f",         
#     cmap="YlGnBu",     
#     linewidths=.5,     
#     cbar_kws={'label': 'V de Cramer (Force de l\'Association)'}
# )
# plt.title('Matrice d\'Association (V de Cramer) sur Données Nettoyées')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

# # Affichage de la matrice numérique
# print("\nMatrice de V de Cramer (Nettoyée):")
# print(cramer_matrix_cleaned)

# import pandas as pd
# import numpy as np

# # --- 1. PRÉPARATION INITIALE (Assurez-vous que ces étapes sont exécutées) ---
# # Vous avez déjà chargé les données précédemment, nous récréons 'Creation Year'
# # au cas où le DataFrame ait été réinitialisé.

# # Assurez-vous que le DataFrame 'metadata' est chargé et que 'Y' est ajouté.
# # metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
# # data = np.load("inf-8245-fall-2025/train.npz")
# # metadata['Y'] = data['y_train'] 

# # Création de la variable 'Creation Year'
# metadata['Create date'] = pd.to_datetime(metadata['Create date'], errors='coerce')
# metadata['Creation Year'] = metadata['Create date'].dt.year


# # --- 2. ANALYSE DES DONNÉES MANQUANTES POUR LES VARIABLES SÉLECTIONNÉES ---

# # Les 3 variables que vous avez conservées, plus la variable cible 'Y'
# cols_to_check = ['Creation Year', 'Location', 'Isolation source', 'Y']

# # 2.1. Calcul des statistiques de NaN
# total_rows = len(metadata)
# missing_stats = metadata[cols_to_check].isnull().sum()
# missing_percentage = (missing_stats / total_rows) * 100

# print("--- Statistiques des Données Manquantes (NaN) pour les variables conservées ---")
# print(f"Nombre total d'observations : {total_rows}\n")

# for col in cols_to_check:
#     print(f"--- Analyse des données manquantes pour la colonne '{col}' ---")
#     print(f"Nombre de valeurs manquantes (NaN) : {missing_stats[col]}")
#     print(f"Taux de données manquantes : {missing_percentage[col]:.2f}%\n")


# # --- 3. FILTRAGE ET AFFICHAGE DES LIGNES INCOMPLÈTES ---

# # Crée un masque booléen: VRAI si la ligne a AU MOINS UN NaN dans les colonnes choisies
# missing_mask_all = metadata[cols_to_check].isnull().any(axis=1)

# # Filtre le DataFrame pour n'afficher que ces lignes
# incomplete_individuals = metadata[missing_mask_all]

# print("--- Aperçu des 5 premières lignes manquantes (si au moins une des 4 colonnes manque) ---")

# # Affichage des colonnes importantes, y compris l'ID pour référence
# cols_to_display = ['ID'] + cols_to_check
# pd.set_option('display.max_columns', None)

# print(incomplete_individuals[cols_to_display].head())

# import pandas as pd
# import numpy as np

# # --- 1. PRÉPARATION DES DONNÉES (Répétition pour garantir la présence de 'Creation Year') ---

# # Si vous avez réinitialisé votre DataFrame, assurez-vous de recharger 'Y' et de créer 'Creation Year'
# # metadata['Y'] = data['y_train'] 
# metadata['Create date'] = pd.to_datetime(metadata['Create date'], errors='coerce')
# metadata['Creation Year'] = metadata['Create date'].dt.year


# # --- 2. FILTRAGE ET AFFICHAGE DE TOUTES LES LIGNES INCOMPLÈTES ---

# # Les 4 colonnes à vérifier
# cols_to_check = ['Creation Year', 'Location', 'Isolation source', 'Y']

# # Crée un masque booléen qui est VRAI si la ligne a AU MOINS UN NaN
# missing_mask_all = metadata[cols_to_check].isnull().any(axis=1)

# # Filtre le DataFrame pour n'afficher que ces lignes
# all_incomplete_individuals = metadata[missing_mask_all]

# # Colonnes à afficher pour l'inspection
# cols_to_display = ['ID'] + cols_to_check

# # --- 3. RÉSULTATS ---

# num_missing = len(all_incomplete_individuals)
# total_rows = len(metadata)

# print(f"--- Individus manquant une valeur dans [{', '.join(cols_to_check)}] ---")
# print(f"Nombre total d'individus dans le jeu de données : {total_rows}")
# print(f"Nombre total d'individus INCOMPLETS : {num_missing}")
# print(f"Pourcentage d'individus incomplets : {(num_missing / total_rows) * 100:.2f}%\n")

# print("--- Affichage de TOUS les individus INCOMPLETS (Colonnes sélectionnées) ---")

# # Configure Pandas pour afficher toutes les lignes
# pd.set_option('display.max_rows', None)  
# pd.set_option('display.max_columns', None)

# # Affichage du DataFrame complet des lignes manquantes
# print(all_incomplete_individuals[cols_to_display])

# # Réinitialiser les options d'affichage de Pandas après l'opération (bonne pratique)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import chi2_contingency

# # --- 1. PRÉPARATION DES DONNÉES ET FONCTION V DE CRAMER ---

# # Assurez-vous d'avoir les données chargées comme suit:
# # metadata = pd.read_csv("inf-8245-fall-2025/metadata_train.csv")
# # data = np.load("inf-8245-fall-2025/train.npz")
# # metadata['Y'] = data['y_train'] 
# # Conversion et création de la variable temporelle catégorielle
# metadata['Create date'] = pd.to_datetime(metadata['Create date'], errors='coerce')
# metadata['Creation Year'] = metadata['Create date'].dt.year

def cramers_v(x, y):
    """Calcule le V de Cramer entre deux variables catégorielles."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi_max = min(r - 1, k - 1)
    if phi_max == 0:
        return 0.0 
    v = np.sqrt(chi2 / (n * phi_max))
    return v

# Liste des variables catégorielles à analyser (excluant ID et la date originale)
cols_to_analyze = [
    'Organism group',
    'Isolation type',
    'Location',
    'Isolation source',
    'Laboratory typing platform',
    'Testing standard',
    'Creation Year',
    'Y' # Variable cible
]

metadata_cat = metadata[cols_to_analyze].copy()


# # --- 2. FONCTION DE REGROUPEMENT EN TOP K ---

def group_rare_categories(series, k=5):
    """Regroupe toutes les catégories sauf les k plus fréquentes en 'Other/Rare'."""
    # Identifier les k catégories les plus fréquentes
    top_k_values = series.value_counts().nlargest(k).index.tolist()
    
    # Créer la nouvelle série
    new_series = series.apply(lambda x: x if x in top_k_values else None)
    return new_series


# # --- 3. APPLICATION DU REGROUPEMENT ET NETTOYAGE ---

metadata_grouped = pd.DataFrame()

# Appliquer le regroupement à toutes les colonnes (sauf Y qui est généralement binaire)
for col in cols_to_analyze:
    if col != 'Y':
        # Appliquer la fonction de regroupement des top 3
        metadata_grouped[col] = group_rare_categories(metadata_cat[col], k=3)
    else:
        # Conserver la variable cible Y telle quelle
        metadata_grouped[col] = metadata_cat[col]

# Afficher le nombre de catégories avant/après pour vérification
print("--- Vérification du nombre de catégories après regroupement ---")
for col in cols_to_analyze:
     print(f"Variable '{col}': {metadata_cat[col].nunique()} -> {metadata_grouped[col].nunique()} catégories")

# # Nettoyage des NaN pour s'assurer que le calcul du V de Cramer fonctionne
# # Le regroupement peut introduire 'Other/Rare' pour les NaNs, mais le dropna assure la cohérence
metadata_final = metadata_grouped.dropna(axis=0, how='any')

# print(f"\nNombre de lignes conservées pour la matrice : {len(metadata_final)}")


# # --- 4. CALCUL ET VISUALISATION DE LA MATRICE (V de Cramer) ---

cols = metadata_final.columns.tolist()
n_cols = len(cols)
cramer_matrix_grouped = pd.DataFrame(index=cols, columns=cols, dtype=float)

# Remplissage de la matrice
for i in range(n_cols):
    for j in range(n_cols):
        var1 = cols[i]
        var2 = cols[j]
        
        if i == j:
            cramer_matrix_grouped.loc[var1, var2] = 1.0
        elif i < j:
            v_score = cramers_v(metadata_final[var1], metadata_final[var2])
            cramer_matrix_grouped.loc[var1, var2] = v_score
            cramer_matrix_grouped.loc[var2, var1] = v_score

# Visualisation
plt.figure(figsize=(12, 10))
sns.heatmap(
    cramer_matrix_grouped,
    annot=True,         
    fmt=".2f",          
    cmap="YlGnBu",     
    linewidths=.5,     
    cbar_kws={'label': 'V de Cramer (Force de l\'Association - Top 5 Groupé)'}
)
plt.title('Matrice d\'Association (V de Cramer) sur Données Simplifiées (Top 5)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nMatrice de V de Cramer (Top 5 Regroupé):")
print(cramer_matrix_grouped)