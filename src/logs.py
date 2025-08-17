import pandas as pd
import os
import datetime

def log_results_csv(
    filename: str = "training_logs.csv",
    model_name: str = "",
    acc: float = 0.0,
    report: dict = None,
    train_size: int = 0,
    test_size: int = 0,
    params: dict = None,
    notes: str = ""
):

    if report is None:
        report = {}

    # Tryin to pull average metrics safely
    macro = report.get("macro avg", {})
    precision = float(macro.get("precision", 0.0))
    recall = float(macro.get("recall", 0.0))
    f1 = float(macro.get("f1-score", 0.0))

    # For getting the Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # To Create new row
    new_entry = pd.DataFrame([[
        None,    
        timestamp,
        model_name,
        float(acc),
        precision,
        recall,
        f1,
        int(train_size),
        int(test_size),
        str(params) if params is not None else "",
        notes
    ]], columns=[
        "RunID","Timestamp","Model","Accuracy","Precision","Recall","F1-score",
        "Train Size","Test Size","Parameters","Notes"
    ])

    # Ensure directory exists if a path is provided
    dirpart = os.path.dirname(filename)
    if dirpart and not os.path.exists(dirpart):
        os.makedirs(dirpart, exist_ok=True)

    # Appending or creating
    if os.path.exists(filename):
        old = pd.read_csv(filename)
        new_entry["RunID"] = len(old) + 1
        final = pd.concat([old, new_entry], ignore_index=True)
    else:
        new_entry["RunID"] = 1
        final = new_entry

    final.to_csv(filename, index=False)
    print(f"Logged run #{int(new_entry['RunID'].iloc[0])} to {filename}")
