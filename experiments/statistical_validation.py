import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from datasets.differential_dataset import generate_dataset as generate_differential
from datasets.integral_dataset import generate_dataset as generate_integral
from datasets.cube_dataset import generate_dataset as generate_cube

ROUNDS = [2, 3, 4, 5]
REPEATS = 10

GENERATORS = {
    "Differential": generate_differential,
    "Integral": generate_integral,
    "Cube": generate_cube,
}


def evaluate_dataset(dataset_path, seed):
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
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
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return accuracy_score(y_test, pred)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    rows = []

    for feature_name, generator in GENERATORS.items():
        print(f"\\n=== {feature_name} ===")

        for r in ROUNDS:
            path = f"results/stat_{feature_name.lower()}_r{r}.csv"

            generator(
                rounds=r,
                samples=5000,
                output_file=path,
            )

            scores = []

            for seed in range(REPEATS):
                acc = evaluate_dataset(path, seed)
                scores.append(acc)

            scores = np.array(scores)

            mean = scores.mean()
            std = scores.std(ddof=1)
            ci95 = 1.96 * std / np.sqrt(REPEATS)

            rows.append({
                "Feature": feature_name,
                "Rounds": r,
                "Mean Accuracy": mean,
                "Std": std,
                "95% CI": ci95,
            })

            print(
                f"Round {r}: {mean:.4f} ± {std:.4f} "
                f"(CI ± {ci95:.4f})"
            )

    results = pd.DataFrame(rows)

    results.to_csv("results/statistical_results.csv", index=False)

    plt.figure(figsize=(7,4))

    for feature in results["Feature"].unique():
        subset = results[results["Feature"] == feature]

        plt.errorbar(
            subset["Rounds"],
            subset["Mean Accuracy"],
            yerr=subset["95% CI"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=feature,
        )

    plt.xticks(ROUNDS)
    plt.ylim(0.45, 1.02)
    plt.xlabel("ASCON Rounds")
    plt.ylabel("XGBoost Accuracy")
    plt.title("Statistically Validated Distinguishability Across ASCON Rounds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/rounds_comparison_errorbars.png", dpi=300)

    print("\\nSaved results/statistical_results.csv")
    print("Saved results/rounds_comparison_errorbars.png")