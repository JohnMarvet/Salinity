from __future__ import annotations

from preprocessing import fit_transform_train, transform_split
from sampling import apply_sampling


def train_model(
    model,
    X_train,
    y_train,
    *,
    sampling: str = "smote",
    normalize: bool = False,
    random_state: int = 42,
    smote_k_neighbors: int | None = None,
):
    X_fit, scaler = fit_transform_train(X_train, normalize)
    X_res, y_res = apply_sampling(
        X_fit,
        y_train,
        method=sampling,
        random_state=random_state,
        k_neighbors=smote_k_neighbors,
    )
    model.fit(X_res, y_res)
    model._salinity_scaler_ = scaler  # type: ignore[attr-defined]
    return model


def transform_for_model(model, X):
    scaler = getattr(model, "_salinity_scaler_", None)
    return transform_split(X, scaler)
