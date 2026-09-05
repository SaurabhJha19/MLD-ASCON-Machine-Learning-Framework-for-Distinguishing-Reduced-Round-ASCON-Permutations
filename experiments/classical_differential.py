import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def classical_differential_score(X):


    return X.sum(axis=1)


def evaluate_classical(dataset_path):

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["label"])
    y = df["label"].values

    scores = classical_differential_score(X)

    class0_mean = scores[y == 0].mean()
    class1_mean = scores[y == 1].mean()

    threshold = (class0_mean + class1_mean) / 2

    if class1_mean > class0_mean:
        predictions = (scores >= threshold).astype(int)
    else:
        predictions = (scores < threshold).astype(int)

    accuracy = accuracy_score(y, predictions)

    return {
        "dataset": dataset_path,
        "classical_accuracy": accuracy,
        "class0_mean_hw": class0_mean,
        "class1_mean_hw": class1_mean,
        "threshold": threshold,
    }


if __name__ == "__main__":

    datasets = [
        "results/differential_r2.csv",
        "results/differential_r3.csv",
        "results/differential_r4.csv",
        "results/differential_r5.csv",
    ]

    results = []

    for dataset in datasets:

        result = evaluate_classical(dataset)

        results.append(result)

        print(
            f"{dataset}: "
            f"Accuracy = {result['classical_accuracy']:.4f}, "
            f"Class0 HW = {result['class0_mean_hw']:.2f}, "
            f"Class1 HW = {result['class1_mean_hw']:.2f}"
        )

    pd.DataFrame(results).to_csv(
        "results/classical_differential.csv",
        index=False
    )

    print(
        "\nSaved to results/classical_differential.csv"
    )