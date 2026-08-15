import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


DATASETS = {
    "Raw": "results/random_vs_ascon_r4.csv",
    "Differential": "results/differential_r4.csv",
    "Intermediate R1": "results/intermediate_round1.csv",
    "Intermediate R2": "results/intermediate_round2.csv",
    "Intermediate R3": "results/intermediate_round3.csv",
    "Intermediate R4": "results/intermediate_round4.csv",
}


def evaluate_dataset(dataset_path):
    df = pd.read_csv(dataset_path)

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
    names = []
    accuracies = []

    print("Feature Representation Comparison (XGBoost)\\n")

    for name, path in DATASETS.items():
        acc = evaluate_dataset(path)
        names.append(name)
        accuracies.append(acc)

        print(f"{name:16s}: {acc:.4f}")

    plt.figure(figsize=(8,4))
    plt.bar(names, accuracies)
    plt.ylim(0.45, 1.02)
    plt.ylabel("Accuracy")
    plt.title("XGBoost Accuracy Across Feature Representations")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig("results/feature_comparison.png", dpi=300)
    print("\\nSaved to results/feature_comparison.png")

    plt.show()