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

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

