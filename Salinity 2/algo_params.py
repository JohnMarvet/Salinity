"""
Hyperparameter definitions and model factory for all supported algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from config import ALGORITHM_CHOICES


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: str  # int, float, bool, choice, optional_int
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    help_text: str = ""


CLASS_WEIGHT_CHOICES = ["balanced", "none"]
MAX_FEATURES_CHOICES = ["sqrt", "log2", "all"]
KERNEL_CHOICES = ["rbf", "linear", "poly"]
PENALTY_CHOICES = ["l2", "l1", "elasticnet"]
KNN_WEIGHTS = ["uniform", "distance"]
SOLVER_CHOICES = ["lbfgs", "liblinear", "saga"]


def _specs(*specs: ParamSpec) -> List[ParamSpec]:
    return list(specs)


ALGO_PARAM_SCHEMAS: Dict[str, List[ParamSpec]] = {
    "random_forest": _specs(
        ParamSpec("n_estimators", "Trees (n_estimators)", "int", 200, 10, 1000, 10),
        ParamSpec(
            "max_depth",
            "Max depth (0 = unlimited)",
            "optional_int",
            8,
            0,
            50,
            1,
            help_text="Limit depth to reduce overfitting on small data.",
        ),
        ParamSpec("min_samples_split", "Min samples split", "int", 2, 2, 50, 1),
        ParamSpec("min_samples_leaf", "Min samples leaf", "int", 2, 1, 50, 1),
        ParamSpec(
            "max_features",
            "Max features per split",
            "choice",
            "sqrt",
            choices=MAX_FEATURES_CHOICES,
        ),
        ParamSpec(
            "class_weight",
            "Class weight",
            "choice",
            "balanced",
            choices=CLASS_WEIGHT_CHOICES,
        ),
        ParamSpec("bootstrap", "Bootstrap samples", "bool", True),
    ),
    "extra_trees": _specs(
        ParamSpec("n_estimators", "Trees (n_estimators)", "int", 200, 10, 1000, 10),
        ParamSpec("max_depth", "Max depth (0 = unlimited)", "optional_int", 8, 0, 50, 1),
        ParamSpec("min_samples_split", "Min samples split", "int", 2, 2, 50, 1),
        ParamSpec("min_samples_leaf", "Min samples leaf", "int", 2, 1, 50, 1),
        ParamSpec(
            "max_features",
            "Max features per split",
            "choice",
            "sqrt",
            choices=MAX_FEATURES_CHOICES,
        ),
        ParamSpec(
            "class_weight",
            "Class weight",
            "choice",
            "balanced",
            choices=CLASS_WEIGHT_CHOICES,
        ),
        ParamSpec("bootstrap", "Bootstrap samples", "bool", False),
    ),
    "gradient_boosting": _specs(
        ParamSpec("n_estimators", "Boosting stages", "int", 120, 10, 500, 10),
        ParamSpec("max_depth", "Max depth", "int", 3, 1, 15, 1),
        ParamSpec("learning_rate", "Learning rate", "float", 0.05, 0.001, 1.0, 0.01),
        ParamSpec("min_samples_split", "Min samples split", "int", 2, 2, 50, 1),
        ParamSpec("min_samples_leaf", "Min samples leaf", "int", 1, 1, 50, 1),
        ParamSpec("subsample", "Subsample", "float", 1.0, 0.3, 1.0, 0.05),
    ),
    "logistic_regression": _specs(
        ParamSpec("C", "Regularization (C)", "float", 1.0, 0.001, 100.0, 0.01),
        ParamSpec("penalty", "Penalty", "choice", "l2", choices=PENALTY_CHOICES),
        ParamSpec("solver", "Solver", "choice", "lbfgs", choices=SOLVER_CHOICES),
        ParamSpec("max_iter", "Max iterations", "int", 2000, 100, 10000, 100),
        ParamSpec(
            "class_weight",
            "Class weight",
            "choice",
            "balanced",
            choices=CLASS_WEIGHT_CHOICES,
        ),
    ),
    "svm_rbf": _specs(
        ParamSpec("C", "Regularization (C)", "float", 1.0, 0.01, 100.0, 0.01),
        ParamSpec("kernel", "Kernel", "choice", "rbf", choices=KERNEL_CHOICES),
        ParamSpec("gamma", "Gamma", "choice", "scale", choices=["scale", "auto"]),
        ParamSpec(
            "class_weight",
            "Class weight",
            "choice",
            "balanced",
            choices=CLASS_WEIGHT_CHOICES,
        ),
    ),
    "knn": _specs(
        ParamSpec("n_neighbors", "Neighbors (k)", "int", 5, 1, 30, 1),
        ParamSpec("weights", "Weighting", "choice", "distance", choices=KNN_WEIGHTS),
        ParamSpec("p", "Minkowski p (1=Manhattan, 2=Euclidean)", "int", 2, 1, 3, 1),
    ),
    "naive_bayes": _specs(
        ParamSpec("var_smoothing", "Variance smoothing", "float", 1e-9, 1e-12, 1e-6, None),
    ),
}


def get_param_schema(algorithm: str) -> List[ParamSpec]:
    if algorithm not in ALGO_PARAM_SCHEMAS:
        raise ValueError(f"No schema for algorithm: {algorithm}")
    return ALGO_PARAM_SCHEMAS[algorithm]


def default_params(algorithm: str) -> Dict[str, Any]:
    return {spec.name: spec.default for spec in get_param_schema(algorithm)}


def _parse_class_weight(value: str):
    return None if value == "none" else value


def _parse_max_features(value: str):
    return None if value == "all" else value


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    iv = int(value)
    return None if iv <= 0 else iv


def build_model(algorithm: str, params: Dict[str, Any], random_state: int = 42):
    p = {**default_params(algorithm), **(params or {})}

    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=_parse_optional_int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=_parse_max_features(p["max_features"]),
            class_weight=_parse_class_weight(p["class_weight"]),
            bootstrap=bool(p["bootstrap"]),
            random_state=random_state,
            n_jobs=-1,
        )

    if algorithm == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=_parse_optional_int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=_parse_max_features(p["max_features"]),
            class_weight=_parse_class_weight(p["class_weight"]),
            bootstrap=bool(p["bootstrap"]),
            random_state=random_state,
            n_jobs=-1,
        )

    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            learning_rate=float(p["learning_rate"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            subsample=float(p["subsample"]),
            random_state=random_state,
        )

    if algorithm == "logistic_regression":
        penalty = p["penalty"]
        solver = p["solver"]
        if penalty == "l1" and solver not in ("liblinear", "saga"):
            solver = "saga"
        if penalty == "elasticnet":
            solver = "saga"
        kwargs = dict(
            C=float(p["C"]),
            penalty=penalty,
            solver=solver,
            max_iter=int(p["max_iter"]),
            class_weight=_parse_class_weight(p["class_weight"]),
            random_state=random_state,
        )
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = 0.5
        return LogisticRegression(**kwargs)

    if algorithm == "svm_rbf":
        return SVC(
            C=float(p["C"]),
            kernel=p["kernel"],
            gamma=p["gamma"],
            class_weight=_parse_class_weight(p["class_weight"]),
            probability=True,
            random_state=random_state,
        )

    if algorithm == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(p["n_neighbors"]),
            weights=p["weights"],
            p=int(p["p"]),
        )

    if algorithm == "naive_bayes":
        return GaussianNB(var_smoothing=float(p["var_smoothing"]))

    supported = ", ".join(ALGORITHM_CHOICES)
    raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {supported}")
