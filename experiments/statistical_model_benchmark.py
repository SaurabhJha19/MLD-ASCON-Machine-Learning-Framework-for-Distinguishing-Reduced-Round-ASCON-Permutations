import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input
from tensorflow.keras.utils import set_random_seed

REPEATS = 10

DATASETS = {
    "Differential R4": "results/differential_r4.csv",
    "Differential R5": "results/tmp_differential_r5.csv",
    "Integral R5": "results/tmp_integral_r5.csv",
    "Cube R5": "results/tmp_cube_r5.csv",
    "Intermediate R4": "results/intermediate_round4.csv",
}


def build_mlp():
    model = Sequential([
        Input(shape=(320,)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_cnn():
    model = Sequential([
        Input(shape=(320, 1)),
        Conv1D(32, 3, activation="relu"),
        Conv1D(64, 3, activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate(dataset_path, model_name, seed):
    set_random_seed(seed)

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    elif model_name == "XGBoost":
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

    elif model_name == "MLP":
        model = build_mlp()
        model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=64,
            verbose=0,
        )
        pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()

    elif model_name == "CNN":
        X_train = X_train.reshape(-1, 320, 1)
        X_test = X_test.reshape(-1, 320, 1)

        model = build_cnn()
        model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=64,
            verbose=0,
        )
        pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()

    else:
        raise ValueError(model_name)

    return accuracy_score(y_test, pred)


if __name__ == "__main__":
    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "MLP",
        "CNN",
    ]

    rows = []

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\\n=== {dataset_name} ===")

        for model_name in model_names:
            scores = []

            for seed in range(REPEATS):
                acc = evaluate(dataset_path, model_name, seed)
                scores.append(acc)

            scores = np.array(scores)

            mean = scores.mean()
            std = scores.std(ddof=1)

            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Mean Accuracy": mean,
                "Std": std,
            })

            print(f"{model_name:20s}: {mean:.4f} ± {std:.4f}")

    results = pd.DataFrame(rows)

    results.to_csv(
        "results/statistical_model_benchmark.csv",
        index=False,
    )

    print("\\nSaved results/statistical_model_benchmark.csv")