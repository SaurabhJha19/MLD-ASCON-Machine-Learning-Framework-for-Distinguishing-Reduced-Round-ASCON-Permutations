import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATASET = "results/integral_r4.csv"
OUTPUT = "results/shap_integral_r4.png"

df = pd.read_csv(DATASET)

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print(f"Accuracy: {accuracy:.4f}")

explainer = shap.TreeExplainer(model)

X_sample = X_test.iloc[:500]

shap_values = explainer.shap_values(X_sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": mean_abs_shap
})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\nTop 20 important bits:")

print(importance.head(20).to_string(index=False))

word_importance = {}

for word in range(5):

    start = word * 64
    end = start + 64

    word_importance[f"x{word}"] = (
        mean_abs_shap[start:end].mean()
    )


print("\nWord-level SHAP importance:")

for word, value in word_importance.items():
    print(f"{word}: {value:.8f}")


plt.figure(figsize=(8, 5))

plt.bar(
    word_importance.keys(),
    word_importance.values()
)

plt.xlabel("ASCON Word")
plt.ylabel("Mean |SHAP value|")
plt.title("SHAP Word-Level Importance — Integral R4")

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300
)

plt.show()

print(f"\nSaved to {OUTPUT}")