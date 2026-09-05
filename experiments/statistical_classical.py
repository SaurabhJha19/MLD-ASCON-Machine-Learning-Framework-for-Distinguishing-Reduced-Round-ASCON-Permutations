import numpy as np
import pandas as pd
from pathlib import Path


DATASETS = {
    2: "results/differential_r2.csv",
    3: "results/differential_r3.csv",
    4: "results/differential_r4.csv",
    5: "results/differential_r5.csv",
}

REPEATS = 10


def evaluate_hamming_weight(df, rng):

    labels = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy()

    scores = X.sum(axis=1)

    indices = np.arange(len(labels))
    rng.shuffle(indices)

    split = int(0.8 * len(indices))

    test_idx = indices[split:]

    test_scores = scores[test_idx]
    test_labels = labels[test_idx]

    train_idx = indices[:split]

    train_scores = scores[train_idx]
    train_labels = labels[train_idx]

    class0_mean = train_scores[train_labels == 0].mean()
    class1_mean = train_scores[train_labels == 1].mean()

    threshold = (class0_mean + class1_mean) / 2

    if class1_mean > class0_mean:
        predictions = (test_scores >= threshold).astype(int)
    else:
        predictions = (test_scores < threshold).astype(int)

    accuracy = np.mean(predictions == test_labels)

    return accuracy


def confidence_interval(values):

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    ci = 1.96 * std / np.sqrt(len(values))

    return mean, std, ci


results = []


for round_no, dataset_path in DATASETS.items():

    df = pd.read_csv(dataset_path)

    accuracies = []

    for seed in range(REPEATS):

        rng = np.random.default_rng(seed)

        accuracy = evaluate_hamming_weight(df, rng)

        accuracies.append(accuracy)

    mean, std, ci = confidence_interval(accuracies)

    results.append({
        "round": round_no,
        "mean_accuracy": mean,
        "std": std,
        "ci_95": ci,
    })

    print(
        f"Round {round_no}: "
        f"{mean:.4f} ± {std:.4f} "
        f"(CI ± {ci:.4f})"
    )


results_df = pd.DataFrame(results)

Path("results").mkdir(exist_ok=True)

results_df.to_csv(
    "results/statistical_classical.csv",
    index=False
)

print("\nSaved to results/statistical_classical.csv")