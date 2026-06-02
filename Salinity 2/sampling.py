from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE


def apply_sampling(
    X_train,
    y_train,
    method: str = "smote",
    random_state: int = 42,
    k_neighbors: int | None = None,
):
    """Resample training data. Returns (X, y) unchanged when method is 'none'."""
    if method == "none":
        return X_train, y_train

    minority = int(np.bincount(y_train.astype(int)).min()) if len(y_train) else 0
    if minority < 2:
        return X_train, y_train

    k = k_neighbors if k_neighbors is not None else min(5, minority - 1)
    k = max(1, min(k, minority - 1))

    if method == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=k)
    elif method == "adasyn":
        sampler = ADASYN(random_state=random_state, n_neighbors=k)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return sampler.fit_resample(X_train, y_train)
