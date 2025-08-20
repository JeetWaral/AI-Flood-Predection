from preprocessing_script import load_data, preprocess_data
from randomforest_model import train_random_forest, evaluate_model as eval_rf
from svm_model import train_svm, evaluate_model as eval_svm
from logs import log_results_csv
from randomforest_model import save_model
from xgboost_model import train_xgboost, evaluate_model as eval_xgb


def main():
    print("-------------------------Started-----------------------")
    file_path = "data/flood_risk_dataset_india.csv"
    print("Loading and Cleaning the data.")
    df = load_data(file_path)

    print(df['Flood Occurred'].value_counts(normalize=True))

    print("Processing data (Encoding + Scaling + Train-Test Split)")
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(df)
    
    ########## MODEL SELECTION ##########
    
    # ======= Random Forest =======
    # model, model_name = train_random_forest(X_train, y_train)
    # acc, report_dict, cm = eval_rf(model, X_test, y_test)
    # notes = "Baseline RF Model"

    # ======= Support Vector Machine =======
    #model, model_name = train_svm(X_train, y_train)
    #acc, report_dict, cm = eval_svm(model, X_test, y_test)
    #notes = "SVM with RBF kernel"

    # ======= XGBoost =======
    model, model_name = train_xgboost(X_train, y_train)
    # Re-load original dataframe for X_test, because your pipeline handles raw data
    df = load_data(file_path)
    _, X_test_raw, _, y_test, _, _ = preprocess_data(df)

    acc, report_dict, cm = eval_xgb(model, X_test_raw, y_test)

    notes = "XGBoost with feature engineering"


    ########## LOGGING + SAVING ##########
    log_results_csv(
        filename="training_logs.csv",
        model_name=model_name,
        acc=acc,
        report=report_dict,
        train_size=len(y_train),
        test_size=len(y_test),
        notes=notes
    )

    save_model(model, scaler, encoders, model_file=f"{model_name.replace(' ', '_')}_model.pkl")
    print(f"Run completed. Accuracy: {acc:.4f}. Logged to training_logs.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
