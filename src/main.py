from sklearn.metrics import classification_report, confusion_matrix
from preprocessing_script import load_data, preprocess_data, interpret_distribution
from randomforest_model import train_random_forest, evaluate_model, save_model
from logs import log_results_csv 

def main():
    # Loading and cleaning the data usning the function created in preporcessing_script.py
    print("-------------------------Started-----------------------")
    file_path = "data/flood_risk_dataset_india.csv"
    print("Loading and Cleaning the data. ")
    df = load_data(file_path)

    #Interpretting the data
    interpret_distribution(df)
    print(df['Flood Occurred'].value_counts(normalize=True))

    print("More Processing on the data, like traing and test data/ Converting Categorical to Numerical.....")
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(df)

    print("Model Started training")
    model, model_name, params = train_random_forest(X_train, y_train)

    # Evaluating The model
    acc, report_dict, cm = evaluate_model(model, X_test, y_test)

    # Log to CSV automatically (one row per run)
    log_results_csv(
        filename="training_logs.csv",
        model_name=model_name,
        acc=acc,
        report=report_dict,
        train_size=len(y_train),
        test_size=len(y_test),
        params=params,
        notes="Baseline RF"
    )

    # Saving th model bundle for later use
    save_model(model, scaler, encoders, model_file="flood_model.pkl")

    print(f"Run completed. Accuracy: {acc:.4f}. Logged to training_logs.csv")

if __name__ == "__main__":
    try: 
        main()
    except Exception as e:
        print(f"Error: {e}")  
