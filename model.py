"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
_BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def _fetch(fname):
    path = os.path.join(tempfile.gettempdir(), fname)

    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(_BASE_URL + fname, tmp)
        os.replace(tmp, path)

    return path


def _read(fname, offset):
    with gzip.open(_fetch(fname), "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


def load_fashion_mnist(n_train=10000, n_test=2000):
    def images(key, n):
        x = _read(_FILES[key], offset=16).reshape(-1, 28, 28)[:n]
        return torch.from_numpy(x.astype(np.float32) / 255.0)

    def labels(key, n):
        y = _read(_FILES[key], offset=8)[:n]
        return torch.from_numpy(y.astype(np.int64))

    return {
        "X_train": images("train_images", n_train),
        "y_train": labels("train_labels", n_train),
        "X_test": images("test_images", n_test),
        "y_test": labels("test_labels", n_test),
    }

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i] - self.mean) / self.std, self.y[i]

# Step 3 - make_loaders
from collections.abc import Mapping

def _unpack(data):
    if isinstance(data, Mapping):
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    if isinstance(data, (tuple, list)):
        if len(data) == 4:
            return tuple(data)
        if len(data) == 2 and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in data):
            (X_tr, y_tr), (X_te, y_te) = data
            return X_tr, y_tr, X_te, y_te

    raise TypeError(
        "make_loaders expected the dict returned by load_fashion_mnist, got "
        f"{type(data).__name__} (shape={getattr(data, 'shape', None)}, len={len(data) if hasattr(data, '__len__') else None})"
    )


def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    X, y, X_test, y_test = _unpack(data)
    n_train = len(X) - val_size

    if n_train <= 0:
        raise ValueError(f"val_size={val_size} leaves no training data (have {len(X)} images)")

    train_ds = FashionDataset(X[:n_train], y[:n_train])
    val_ds = FashionDataset(X[n_train:], y[n_train:])
    test_ds = FashionDataset(X_test, y_test)

    generator = torch.Generator().manual_seed(seed)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "sizes": (len(train_ds), len(val_ds), len(test_ds)),
    }

# Step 4 - MLP
class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.out(x)


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y in loader:
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches else 0.0

# Step 6 - evaluate
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    n_examples = 0

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            batch_size = y.size(0)
            total_loss += loss_fn(logits, y).item() * batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()
            n_examples += batch_size

    if n_examples == 0:
        return 0.0, 0.0

    return total_loss / n_examples, correct / n_examples

# Step 7 - fit
def fit(model, loaders, epochs=5, lr=0.05, seed=42):
    torch.manual_seed(seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, loaders["val"], loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch

    return history

# Step 8 - lr_range_test
import itertools

def lr_range_test(make_model, loader, lrs, n_batches=20, seed=42):
    batches = list(itertools.islice(loader, n_batches))

    if not batches:
        raise ValueError("loader yield no batches")

    loss_fn = nn.CrossEntropyLoss()
    results = {}

    for lr in lrs:
        torch.manual_seed(seed)
        model = make_model()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        total = 0.0

        for X, y in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        results[lr] = total / len(batches)

    return results

# Step 9 - random_search
import torch

def random_search(loaders, n_trials=4, epochs=2, seed=42):
    rng = np.random.default_rng(seed)
    trials = []

    for _ in range(n_trials):
        hidden1 = int(rng.choice([100, 200, 300]))
        hidden2 = int(rng.choice([50, 100]))
        lr = float(rng.choice([0.01, 0.05, 0.1]))

        torch.manual_seed(seed)
        model = MLP(hidden1, hidden2)
        history = fit(model, loaders, epochs, lr, seed)

        trials.append({
            "hidden1": hidden1,
            "hidden2": hidden2,
            "lr": lr,
            "val_acc": max(history["val_acc"]),
        })

    best = max(trials, key=lambda t: t["val_acc"]) if trials else None
    
    return {"trials": trials, "best": best}

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

