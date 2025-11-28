import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

# --- Charger X et y ---
data = np.load("inf-8245-fall-2025/train.npz")
data_test = np.load("inf-8245-fall-2025/test.npz")

X_train = data['X_train']
y_train = data['y_train']
X_test  = data_test['X_test']
ids_test = data_test['ids']

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier



# --- Pipeline SVD + LightGBM ---
pipe = Pipeline([
    ("svd", TruncatedSVD(n_components=400, random_state=42)),
    ("scaler", StandardScaler()),
    ("lgbm", LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42
    ))
])

print("🔄 Entraînement du modèle SVD+LightGBM...")
pipe.fit(X_train, y_train)

print("🔮 Prédiction...")
y_pred = pipe.predict(X_test)

# --- Générer le CSV ---
data_test_kaggle = np.load("inf-8245-fall-2025/test.npz", allow_pickle=True)
ids_test = data_test_kaggle["ids"]

submission = pd.DataFrame({"id": ids_test, "label": y_pred})
submission.to_csv("submission_lightgbm.csv", index=False)

print("✅ submission_lightgbm.csv généré !")
