import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from preprocessing_script import load_and_clean_data, preprocess_features
from logisticreg import LogisticRegressionScratch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  

if __name__ == "__main__":
    try:
        print("-------------------------Started-----------------------")
        filepath = os.path.join("data", "flood_risk_dataset_india.csv")

        # Loading and cleaning the data usning the function created in preporcessing_script.py
        print("Loading and Cleaning the data. ")
        df = load_and_clean_data(filepath)
        print("Data loaded and cleaned Sucess")
        print(df.head())

        print("More Processing on the data, like traing and test data/ Converting Categorical to Numerical.....")
        X_train, X_test, y_train, y_test, encoder, scaler = preprocess_features(df)

        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Flood occurrence distribution in training set:\n{pd.Series(y_train).value_counts(normalize=True)}")

        print("Model Started training")
        model = LogisticRegressionScratch(learning_rate=0.1, epochs=1000)
        model.fit(X_train, y_train)

        print("Model Trained.....")
        y_pred = model.predict(X_test)
        print("y_pred shape:", y_pred.shape)
        print("y_test shape:", y_test.shape)

        accuracy = np.mean(y_pred == y_test.flatten())
        print(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Error: {e}")  
