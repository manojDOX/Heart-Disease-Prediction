import os
import pickle
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = {
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                'n_estimators': [100],
                'max_depth': [3, 5],
                'learning_rate': [0.05],
                'subsample': [0.8]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100],
                'max_depth': [5],
                'min_samples_split': [5]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'max_depth': [3, 5],
                'min_samples_split': [2, 4]
            }
        }
    }

    for name, cfg in models.items():
        print(f"\n🔍 Training {name}...")
        grid = GridSearchCV(cfg["model"], cfg["params"], cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_bal, y_train_bal)
        best_model = grid.best_estimator_

        model_path = os.path.join(model_dir, f"{name.lower()}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"✅ {name} saved to {model_path} with best params: {grid.best_params_}")