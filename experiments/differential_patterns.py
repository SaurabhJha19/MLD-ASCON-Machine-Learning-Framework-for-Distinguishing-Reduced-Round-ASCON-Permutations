import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PATTERNS = {
    "P1 (x0,b0)": "results/diff_p1.csv",
    "P2 (x0,b15)": "results/diff_p2.csv",
    "P3 (x1,b0)": "results/diff_p3.csv",
    "P4 (x2,b31)": "results/diff_p4.csv",
    "P5 (x4,b63)": "results/diff_p5.csv",
}


def evaluate(path):
    df = pd.read_csv(path)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return accuracy_score(y_test, pred)


if __name__ == "__main__":
    rows = []

    print("Differential Pattern Robustness\\n")

    for name, path in PATTERNS.items():
        acc = evaluate(path)

        rows.append({
            "Pattern": name,
            "Accuracy": acc,
        })

        print(f"{name:12s}: {acc:.4f}")

    pd.DataFrame(rows).to_csv(
        "results/differential_patterns.csv",
        index=False,
    )

    print("\\nSaved results/differential_patterns.csv")