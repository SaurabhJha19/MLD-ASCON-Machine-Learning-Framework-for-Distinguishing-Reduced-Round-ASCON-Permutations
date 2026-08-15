import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def train_logistic(dataset_path="results/random_vs_ascon_r4.csv"):
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

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    train_logistic("results/differential_r4.csv")