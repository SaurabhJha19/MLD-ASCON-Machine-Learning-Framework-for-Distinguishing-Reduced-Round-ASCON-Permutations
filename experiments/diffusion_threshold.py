import pandas as pd
import matplotlib.pyplot as plt

hamming_distance = {
    1: 7.51,
    2: 49.25,
    3: 140.62,
    4: 159.98,
}

ml_accuracy = {
    1: 1.0000,
    2: 1.0000,
    3: 0.9998,
    4: 0.5035,
    5: 0.5067,
}

rows = []

for round_no in sorted(ml_accuracy):

    rows.append({
        "round": round_no,
        "hamming_distance": hamming_distance.get(
            round_no, None
        ),
        "ml_accuracy": ml_accuracy[round_no],
        "distinguishable": (
            abs(ml_accuracy[round_no] - 0.5) > 0.05
        )
    })


df = pd.DataFrame(rows)

print("\nDiffusion Threshold Analysis\n")
print(df.to_string(index=False))

threshold_round = None

for _, row in df.iterrows():

    if not row["distinguishable"]:

        threshold_round = int(row["round"])
        break


print(
    f"\nEstimated distinguishability threshold: "
    f"Round {threshold_round}"
)

df.to_csv(
    "results/diffusion_threshold.csv",
    index=False
)

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    df["round"],
    df["hamming_distance"],
    marker="o",
    linewidth=2,
    label="Average Hamming Distance"
)

ax1.set_xlabel("ASCON Rounds")
ax1.set_ylabel("Average Hamming Distance")

ax2 = ax1.twinx()

ax2.plot(
    df["round"],
    df["ml_accuracy"],
    marker="s",
    linewidth=2,
    label="ML Accuracy"
)

ax2.axhline(
    0.5,
    linestyle="--",
    linewidth=1
)

ax2.set_ylabel("XGBoost Accuracy")

plt.title(
    "Diffusion and ML Distinguishability Across ASCON Rounds"
)

fig.tight_layout()

plt.savefig(
    "results/diffusion_threshold.png",
    dpi=300
)

plt.show()

print(
    "\nSaved to results/diffusion_threshold.csv"
)

print(
    "Saved to results/diffusion_threshold.png"
)