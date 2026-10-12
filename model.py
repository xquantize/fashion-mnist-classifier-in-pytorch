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

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

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

