"""Legacy helpers — prefer data.py and pipeline.py."""

from data import prepare_dataset as load_data
from pipeline import RunConfig
from sklearn.model_selection import train_test_split


def split_data(X, y, seed=None, test_size=0.25):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)
