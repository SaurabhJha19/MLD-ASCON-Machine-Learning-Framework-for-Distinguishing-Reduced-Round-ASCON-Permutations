import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv("results/intermediate_round3.csv")

X = df.drop(columns=["label"]).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    max_iter=50,
    random_state=42,
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance = result.importances_mean

word_importance = []
for i in range(5):
    start = i * 64
    end = (i + 1) * 64
    word_importance.append(np.sum(importance[start:end]))

words = ["x0", "x1", "x2", "x3", "x4"]

plt.figure(figsize=(6, 4))
plt.bar(words, word_importance)
plt.xlabel("ASCON Word")
plt.ylabel("Permutation Importance")
plt.title("Word-Level Importance (Intermediate Round 3)")
plt.tight_layout()

plt.savefig("results/word_importance_round3.png", dpi=300)
print("Saved to results/word_importance_round3.png")

plt.show()

for w, val in zip(words, word_importance):
    print(f"{w}: {val:.6f}")