import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATASETS = {
    "Differential": "results/differential_r4.csv",
    "Integral": "results/integral_r4.csv",
    "Cube": "results/cube_r4.csv",
    "Intermediate R3": "results/intermediate_round3.csv",
}

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for dataset_name, dataset_path in DATASETS.items():
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

    for model_name, model in MODELS.items():
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": acc,
            "ROC-AUC": auc,
        })

results_df = pd.DataFrame(results)

print("Model Comparison Results\\n")
print(results_df)

results_df.to_csv("results/model_comparison.csv", index=False)

pivot = results_df.pivot(index="Dataset", columns="Model", values="Accuracy")

pivot.plot(kind="bar", figsize=(8,4))

plt.ylim(0.45, 1.02)
plt.ylabel("Accuracy")
plt.title("Model Comparison Across Feature Representations")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("results/model_comparison.png", dpi=300)

print("\\nSaved results/model_comparison.csv")
print("Saved results/model_comparison.png")