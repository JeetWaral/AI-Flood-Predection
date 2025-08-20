import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_col="Flood Occurred"):

    # Droping the rows with missing values
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encoding the categorical features, Convrting the string/object data to Int
    cat_cols = X.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Scaling the numerical features (Scaling feature columns to get mean 0, standard deviation )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders
