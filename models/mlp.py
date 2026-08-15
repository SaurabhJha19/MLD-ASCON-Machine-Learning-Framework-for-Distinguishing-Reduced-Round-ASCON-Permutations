import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(dataset_path="results/random_vs_ascon_r4.csv", epochs=20, batch_size=128):
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["label"]).values.astype("float32")
    y = df["label"].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test)).squeeze().numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    return model


if __name__ == "__main__":
    train_mlp("results/differential_r4.csv")