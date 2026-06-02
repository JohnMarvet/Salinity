from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def fit_transform_train(X_train, normalize: bool):
    if not normalize:
        return X_train, None
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler


def transform_split(X, scaler):
    if scaler is None:
        return X
    return scaler.transform(X)
