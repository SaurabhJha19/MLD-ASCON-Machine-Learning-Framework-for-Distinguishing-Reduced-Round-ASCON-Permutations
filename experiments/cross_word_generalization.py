import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

TRAIN = "results/integral_x0_r4.csv"

TESTS = {
    "x0": "results/integral_x0_r4.csv",
    "x1": "results/integral_x1_r4.csv",
    "x2": "results/integral_x2_r4.csv",
    "x3": "results/integral_x3_r4.csv",
    "x4": "results/integral_x4_r4.csv",
}


def load(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


if __name__ == "__main__":
    X_train, y_train = load(TRAIN)

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

    rows = []

    print("Train on x0-active Integral R4\\n")

    for name, path in TESTS.items():
        X_test, y_test = load(path)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        rows.append({
            "Test Word": name,
            "Accuracy": acc,
        })

        print(f"x0 -> {name}: {acc:.4f}")

    pd.DataFrame(rows).to_csv(
        "results/cross_word_generalization.csv",
        index=False,
    )

    print("\\nSaved results/cross_word_generalization.csv")