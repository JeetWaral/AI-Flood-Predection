from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from imblearn.over_sampling import SMOTE

# def train_random_forest(X_train, y_train):
      # Trying to train Random Forest by giving  the no of estimators and the no of random states
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)

#     # Geting the feature names safely to avoid failing
#     if hasattr(X_train, 'columns'):   # If DataFrame
#         feature_names = X_train.columns
#     else:
#         feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

#     importances = rf.feature_importances_
#     feature_importances = sorted(zip(importances, feature_names), reverse=True)

#     print("\nTop 10 Important Features:")
#     for importance, name in feature_importances[:10]:
#         print(f"{name}: {importance:.4f}")

#     return rf, "RandomForestClassifier", rf.get_params()

def train_random_forest(X, y):
    # Converting the Columns to DataFrame if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Identify categorical & numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

 
    # Here we are doing Feature Engineering, Example: rainfall * soil absorption, slope * distance_from_river etc, to get more details
    if "rainfall" in X.columns and "soil_absorption" in X.columns:
        X["rainfall_absorption"] = X["rainfall"] * X["soil_absorption"]

    if "slope" in X.columns and "river_distance" in X.columns:
        X["slope_river_interaction"] = X["slope"] * X["river_distance"]

    
    # Using SMOTE (Synthetic Minority Over-sampling Technique) for balancing classes by generating samples for the minority class, Helps in Creating Model Biases
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    #  Preprocesing(SS normalize input for algo, onehot converts categories to numeric vector)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # a pipeline that sequentally preprocesses inputs and trains a balanced Random Forest classifier with 200 trees.
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",  
            random_state=42
        ))
    ])

    # Training begins here
    model.fit(X_resampled, y_resampled)

    return model,"Random Forest"


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy: ", round(acc, 4))
    print("Classification Report:", report_text)
    print("Confusion Matrix:", cm)

    # Also return the structured dict report for logging
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return acc, report_dict, cm


def save_model(model, scaler, label_encoders, model_file="flood_model.pkl"):
    package = {"model": model, "scaler": scaler, "encoders": label_encoders}
    joblib.dump(package, model_file)
    print(f"Model, scaler, and encoders saved to {model_file}")


def load_model(model_file="flood_model.pkl"):
    package = joblib.load(model_file)
    return package["model"], package["scaler"], package["encoders"]
