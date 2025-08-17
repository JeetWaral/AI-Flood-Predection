import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    df.columns = [col.replace("Â", "").replace("³", "3").replace("²", "2").strip()
                  for col in df.columns]

    return df


def preprocess_data(df, target_col="Flood Occurred"):

    # Drop rows with missing values
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encoding the categorical features 
    cat_cols = X.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Scaling the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders


def interpret_distribution(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    for col in numeric_df.columns:
        skew = numeric_df[col].skew()
        kurt = numeric_df[col].kurtosis()
        print(f"\nFeature: {col}")
        if abs(skew) < 0.5:
            print(" - Distribution is fairly symmetric")
        elif skew > 0.5:
            print(" - Positively skewed (long right tail, many small values)")
        else:
            print(" - Negatively skewed (long left tail, many large values)")
        
        if kurt > 3:
            print(" - Heavy-tailed (many outliers)")
        elif kurt < 3:
            print(" - Light-tailed (few outliers)")
        else:
            print(" - Close to normal distribution")

