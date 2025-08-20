from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def train_svm(X, y):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Feature Engineering (optional but helps improve performance)
    if "rainfall" in X.columns and "elevation" in X.columns:
        X["rainfall_elevation_interaction"] = X["rainfall"] * X["elevation"]

    if "humidity" in X.columns and "temperature" in X.columns:
        X["humid_temp_interaction"] = X["humidity"] * X["temperature"]

    # Handling class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Full training pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])

    model.fit(X_resampled, y_resampled)
    return model, "Support Vector Machine"

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy: ", round(acc, 4))
    print("Classification Report:", report_text)
    print("Confusion Matrix:", cm)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return acc, report_dict, cm
