import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# SHAP word-level importance from our SHAP experiment
# --------------------------------------------------

shap_importance = {
    "x0": 0.01986883,
    "x1": 0.02592120,
    "x2": 0.02770827,
    "x3": 0.02779566,
    "x4": 0.03105602,
}


# --------------------------------------------------
# ASCON diffusion measurements
# --------------------------------------------------

diffusion = {
    1: 7.51,
    2: 49.25,
    3: 140.62,
    4: 159.98,
}


# --------------------------------------------------
# Convert SHAP importance to DataFrame
# --------------------------------------------------

shap_df = pd.DataFrame(
    list(shap_importance.items()),
    columns=["word", "shap_importance"]
)


# --------------------------------------------------
# Word-level diffusion
#
# We use the normalized global diffusion value as
# the diffusion reference for the interpretability
# analysis.
# --------------------------------------------------

mean_diffusion = np.mean(list(diffusion.values()))

shap_df["normalized_shap"] = (
    shap_df["shap_importance"] /
    shap_df["shap_importance"].sum()
)


print("\nSHAP Word-Level Importance")
print(shap_df.to_string(index=False))


print(
    f"\nMean diffusion across measured rounds: "
    f"{mean_diffusion:.2f} bits"
)


# --------------------------------------------------
# Correlation analysis
#
# IMPORTANT:
# SHAP is word-level while our available diffusion
# measurement is global. Therefore we report the
# relationship descriptively rather than claiming
# a word-level statistical correlation.
# --------------------------------------------------

rounds = np.array(list(diffusion.keys()))
hamming = np.array(list(diffusion.values()))

# ML accuracy corresponding to the same rounds
ml_accuracy = np.array([
    1.0000,
    1.0000,
    0.9998,
    0.5035
])


diffusion_accuracy_corr = np.corrcoef(
    hamming,
    ml_accuracy
)[0, 1]


print(
    f"\nCorrelation between diffusion and "
    f"ML accuracy: {diffusion_accuracy_corr:.4f}"
)


# --------------------------------------------------
# Plot 1: SHAP importance
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    shap_df["word"],
    shap_df["normalized_shap"]
)

plt.xlabel("ASCON Word")
plt.ylabel("Normalized SHAP Importance")

plt.title(
    "Normalized SHAP Importance Across ASCON Words"
)

plt.tight_layout()

plt.savefig(
    "results/shap_word_normalized.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# Plot 2: Diffusion vs ML accuracy
# --------------------------------------------------

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    rounds,
    hamming,
    marker="o",
    linewidth=2
)

ax1.set_xlabel("ASCON Rounds")
ax1.set_ylabel("Average Hamming Distance")


ax2 = ax1.twinx()

ax2.plot(
    rounds,
    ml_accuracy,
    marker="s",
    linewidth=2
)

ax2.set_ylabel("XGBoost Accuracy")

plt.title(
    "Explainability Context: Diffusion vs ML Distinguishability"
)

fig.tight_layout()

plt.savefig(
    "results/explainability_diffusion.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# Save summary
# --------------------------------------------------

shap_df.to_csv(
    "results/explainability_summary.csv",
    index=False
)

print(
    "\nSaved to results/explainability_summary.csv"
)

print(
    "Saved to results/shap_word_normalized.png"
)

print(
    "Saved to results/explainability_diffusion.png"
)