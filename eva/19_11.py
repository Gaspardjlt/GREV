import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- Charger X et y ---
data_X = np.load("eva/X_reduit_190.npz")
data_y = np.load("eva/y_train_190.npz")
X_train_data = data_X["X"]
y_train = data_y["y"]

data_test_X = np.load("inf-8245-fall-2025/X_test_reduit_190.npz")
X_test_data = data_test_X["X"]

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_data)
X_test_scaled = scaler.transform(X_test_data)

# --- Modèle XGBoost Classifier ---
model = XGBClassifier(
    n_estimators=300,           
    max_depth=6,                
    learning_rate=0.1,          
    subsample=0.8,              
    colsample_bytree=0.1,       
    min_child_weight=3,         
    gamma=0.1,                  
    reg_lambda=1.0,             
    tree_method='hist',         
    n_jobs=-1,                  
    random_state=42,
    verbosity=1
)

# --- Entraînement ---
model.fit(X_train_scaled, y_train)

# --- Prédiction ---
y_pred = model.predict(X_test_scaled)

# --- Submission Kaggle ---
data_test = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
ids_test = data_test["ids"]

submission = pd.DataFrame({"id": ids_test, "label": y_pred})
submission.to_csv("submission_xgb_classifier_no_metadata.csv", index=False)

print("✅ submission_xgb_classifier_no_metadata.csv généré !")
