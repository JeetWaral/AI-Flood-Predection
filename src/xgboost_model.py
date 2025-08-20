import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def train_xgboost(X, y):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Feature Engineering
    if "Rainfall" in X.columns and "Humidity" in X.columns:
        X["Rainfall_Humidity"] = X["Rainfall"] * X["Humidity"]
    if "Temperature" in X.columns and "Elevation" in X.columns:
        X["Temp_Elevation"] = X["Temperature"] * X["Elevation"]
    if "River Discharge" in X.columns and "Water Level" in X.columns:
        X["Discharge_Level"] = X["River Discharge"] * X["Water Level"]
    if "Rainfall" in X.columns and "Elevation" in X.columns:
        X["Rainfall_to_Elevation"] = X["Rainfall"] / (X["Elevation"] + 1)

    # Balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Update numeric and categorical columns after feature engineering
    categorical_cols = X_resampled.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X_resampled.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        ))
    ])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_resampled, y_resampled)

    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    return best_model, "XGBoost (Tuned)"

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy: ", round(acc, 4))
    print("Classification Report:\n", report_text)
    print("Confusion Matrix:\n", cm)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return acc, report_dict, cm
