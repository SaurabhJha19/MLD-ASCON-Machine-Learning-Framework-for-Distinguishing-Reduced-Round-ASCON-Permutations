import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

PAIRS = [
    ("Differential", "results/tmp_differential_r2.csv", "results/tmp_differential_r3.csv"),
    ("Differential", "results/tmp_differential_r3.csv", "results/tmp_differential_r4.csv"),
    ("Differential", "results/tmp_differential_r4.csv", "results/tmp_differential_r5.csv"),
    ("Integral", "results/tmp_integral_r3.csv", "results/tmp_integral_r4.csv"),
    ("Integral", "results/tmp_integral_r4.csv", "results/tmp_integral_r5.csv"),
    ("Cube", "results/tmp_cube_r3.csv", "results/tmp_cube_r4.csv"),
    ("Cube", "results/tmp_cube_r4.csv", "results/tmp_cube_r5.csv"),
]


def load(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


if __name__ == "__main__":
    rows = []

    for feature, train_path, test_path in PAIRS:
        X_train, y_train = load(train_path)
        X_test, y_test = load(test_path)

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

        acc = accuracy_score(y_test, pred)

        train_round = train_path.split("_r")[1].split(".")[0]
        test_round = test_path.split("_r")[1].split(".")[0]

        rows.append({
            "Feature": feature,
            "Train Round": train_round,
            "Test Round": test_round,
            "Accuracy": acc,
        })

        print(
            f"{feature:12s} R{train_round} -> R{test_round}: {acc:.4f}"
        )

    pd.DataFrame(rows).to_csv(
        "results/cross_round_generalization.csv",
        index=False,
    )

    print("\\nSaved results/cross_round_generalization.csv")