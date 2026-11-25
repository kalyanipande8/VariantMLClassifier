"""Real small PyTorch MLP (a genuine deep-learning classifier, distinct
from BiomarkerClassifier's GradientBoostingClassifier) for variant
pathogenicity prediction."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


class VariantMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(X, y, seed: int = 7, epochs: int = 150):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Xt = torch.tensor(X_train.astype(np.float32))
    yt = torch.tensor(y_train.astype(np.int64))

    model = VariantMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test.astype(np.float32)))
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs > 0.5).astype(int)

    return {"accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "auc": round(float(roc_auc_score(y_test, probs)), 4)}
