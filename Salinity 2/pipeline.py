from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RUNS,
    DEFAULT_SALINITY_THRESHOLD,
    DEFAULT_TEST_SIZE,
)
from data import class_distribution, prepare_dataset
from evaluate import MetricBundle, evaluate_model
from models import get_model
from threshold_search import find_best_threshold
from train import train_model, transform_for_model


@dataclass
class RunConfig:
    data_path: Optional[str] = None
    salinity_threshold: float = DEFAULT_SALINITY_THRESHOLD
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD
    drop_columns: Sequence[str] = field(default_factory=list)
    use_columns: Optional[Sequence[str]] = None
    algorithm: str = "random_forest"
    model_params: Dict[str, Any] = field(default_factory=dict)
    sampling: str = "smote"
    smote_k_neighbors: Optional[int] = None
    normalize: bool = True
    n_runs: int = DEFAULT_RUNS
    test_size: float = DEFAULT_TEST_SIZE
    random_seed: Optional[int] = None
    evaluation_mode: str = "holdout"  # holdout | stratified_cv
    cv_folds: int = 5
    auto_tune_threshold: bool = False
    tune_metric: str = "recall"  # recall | f1


@dataclass
class ExperimentResult:
    config: RunConfig
    n_samples: int
    n_features: int
    class_balance: dict
    feature_names: List[str]
    run_metrics: List[MetricBundle]
    summary: Dict[str, float]
    tuned_thresholds: List[float] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None

    def summary_dataframe(self) -> pd.DataFrame:
        rows = [m.as_dict() for m in self.run_metrics]
        return pd.DataFrame(rows)


def _aggregate_metrics(metrics: List[MetricBundle]) -> Dict[str, float]:
    keys = ["accuracy", "balanced_accuracy", "recall", "precision", "f1"]
    out: Dict[str, float] = {}
    for key in keys:
        values = [getattr(m, key) for m in metrics]
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    roc_values = [m.roc_auc for m in metrics if m.roc_auc is not None]
    if roc_values:
        out["roc_auc_mean"] = float(np.mean(roc_values))
        out["roc_auc_std"] = float(np.std(roc_values))
    return out


def _safe_stratified_split(X, y, test_size, seed):
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def _feature_importance(model, feature_names: List[str]) -> Optional[Dict[str, float]]:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return {name: float(v) for name, v in zip(feature_names, vals)}
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).ravel()
        return {name: float(v) for name, v in zip(feature_names, coef)}
    return None


def _train_and_evaluate_fold(
    config: RunConfig,
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names: List[str],
    seed: int,
) -> tuple[MetricBundle, float, Optional[Dict[str, float]]]:
    model = get_model(config.algorithm, random_state=seed, params=config.model_params)
    model = train_model(
        model,
        X_train,
        y_train,
        sampling=config.sampling,
        normalize=config.normalize,
        random_state=seed,
        smote_k_neighbors=config.smote_k_neighbors,
    )

    X_val_t = transform_for_model(model, X_val)
    threshold = config.decision_threshold

    if config.auto_tune_threshold and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val_t)[:, 1]
        threshold, _ = find_best_threshold(
            y_val, proba, metric=config.tune_metric
        )

    metrics = evaluate_model(model, X_val_t, y_val, decision_threshold=threshold)
    importance = _feature_importance(model, feature_names)
    return metrics, threshold, importance


def _run_holdout_once(config: RunConfig, X, y, feature_names, seed) -> tuple:
    X_train, X_val, y_train, y_val = _safe_stratified_split(
        X, y, config.test_size, seed
    )
    return _train_and_evaluate_fold(
        config, X_train, X_val, y_train, y_val, feature_names, seed
    )


def _run_cv_once(config: RunConfig, X, y, feature_names, seed) -> List[tuple]:
    minority = int(y.value_counts().min())
    n_splits = min(config.cv_folds, minority, len(y))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        results.append(
            _train_and_evaluate_fold(
                config, X_train, X_val, y_train, y_val, feature_names, seed
            )
        )
    return results


def run_experiment(
    config: RunConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> ExperimentResult:
    X, y, _meta = prepare_dataset(
        path=config.data_path,
        salinity_threshold=config.salinity_threshold,
        drop_columns=config.drop_columns if not config.use_columns else None,
        use_columns=config.use_columns,
    )

    if y.nunique() < 2:
        raise ValueError(
            f"Only one class present at Ece threshold {config.salinity_threshold}. "
            "Adjust the threshold or add more saline samples."
        )

    minority = int(y.value_counts().min())
    if minority < 2 and config.n_runs > 1 and config.evaluation_mode == "holdout":
        raise ValueError(
            f"Only {minority} sample(s) in the minority class. "
            "Lower the Ece threshold or add more saline profiles."
        )

    feature_names = list(X.columns)
    run_metrics: List[MetricBundle] = []
    tuned_thresholds: List[float] = []
    importances: List[Dict[str, float]] = []
    rng = np.random.default_rng(config.random_seed)

    for i in range(config.n_runs):
        if progress_callback:
            progress_callback(i + 1, config.n_runs)

        seed = (
            int(rng.integers(0, 2**31 - 1))
            if config.random_seed is None
            else config.random_seed + i
        )

        if config.evaluation_mode == "stratified_cv":
            fold_results = _run_cv_once(config, X, y, feature_names, seed)
            for metrics, threshold, importance in fold_results:
                run_metrics.append(metrics)
                tuned_thresholds.append(threshold)
                if importance:
                    importances.append(importance)
        else:
            metrics, threshold, importance = _run_holdout_once(
                config, X, y, feature_names, seed
            )
            run_metrics.append(metrics)
            tuned_thresholds.append(threshold)
            if importance:
                importances.append(importance)

    summary = _aggregate_metrics(run_metrics)

    avg_importance: Optional[Dict[str, float]] = None
    if importances:
        keys = importances[0].keys()
        avg_importance = {
            k: float(np.mean([imp[k] for imp in importances])) for k in keys
        }

    if tuned_thresholds and config.auto_tune_threshold:
        summary["tuned_threshold_mean"] = float(np.mean(tuned_thresholds))
        summary["tuned_threshold_std"] = float(np.std(tuned_thresholds))

    return ExperimentResult(
        config=config,
        n_samples=len(y),
        n_features=X.shape[1],
        class_balance=class_distribution(y),
        feature_names=feature_names,
        run_metrics=run_metrics,
        summary=summary,
        tuned_thresholds=tuned_thresholds,
        feature_importance=avg_importance,
    )


def config_to_dict(config: RunConfig) -> dict:
    return asdict(config)
