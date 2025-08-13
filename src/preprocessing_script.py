import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    # Cleaning the column names 
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
    df.rename(columns={'Temperature_Â°C': 'Temperature_C'}, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

def preprocess_features(df):

    # Separating the target
    X = df.drop('Flood_Occurred', axis=1)
    y = df['Flood_Occurred']

    #Check categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns.to_list()
    num_cols = X.select_dtypes(exclude=['object']).columns.to_list()


     # One-hot Library used for Convering Categorical Columns to Numericals
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if cat_cols:
        cat_encoded = encoder.fit_transform(X[cat_cols])
    else:
        cat_encoded = np.array([]).reshape(len(X), 0)


    # Thn Scalling the Numerical Columns
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[num_cols]) if num_cols else np.array([]).reshape(len(X), 0)
    X_processed = np.hstack([X_num_scaled, cat_encoded])

    #Splitting the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    return X_train, X_test, y_train, y_test, encoder, scaler
