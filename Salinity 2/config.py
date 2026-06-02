from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# First existing path wins (same folder as the scripts).
DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "Data.xlsx",
    PROJECT_ROOT / "data" / "Data.xlsx",
    Path(r"C:\Users\alrhe\Downloads\Telegram Desktop\Data.xlsx"),
]

TARGET_COLUMN = "Ece mean"
LEAKAGE_COLUMNS = ("Ece mean", "Ece_class", "Salinity_Class")
ID_COLUMNS = ("ID1", "Profile_ID")

DEFAULT_SALINITY_THRESHOLD = 3.0
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RUNS = 30
DEFAULT_DECISION_THRESHOLD = 0.5

ALGORITHM_CHOICES = {
    "random_forest": "Random Forest (balanced, good default)",
    "extra_trees": "Extra Trees (fast ensemble)",
    "gradient_boosting": "Gradient Boosting (strong but can overfit small data)",
    "logistic_regression": "Logistic Regression (linear baseline)",
    "svm_rbf": "SVM RBF (works on few samples with scaling)",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes (very small data)",
}

SAMPLING_CHOICES = {
    "none": "No resampling (use class weights where supported)",
    "smote": "SMOTE oversampling",
    "adasyn": "ADASYN (focuses on harder minority points)",
}

EVALUATION_MODES = {
    "holdout": "Repeated random hold-out validation",
    "stratified_cv": "Stratified k-fold cross-validation each run",
}

TUNE_METRICS = ("recall", "f1")
