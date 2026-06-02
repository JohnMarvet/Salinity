"""
Salinity ML Lab — full control over features, training, and algorithm hyperparameters.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from algo_params import default_params
from config import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RUNS,
    DEFAULT_SALINITY_THRESHOLD,
    DEFAULT_TEST_SIZE,
    LEAKAGE_COLUMNS,
    PROJECT_ROOT,
    SAMPLING_CHOICES,
    TARGET_COLUMN,
)
from data import (
    build_labels,
    class_distribution,
    get_selectable_feature_columns,
    load_raw_dataframe,
    prepare_dataset,
    resolve_data_path,
)
from models import list_algorithms
from pipeline import RunConfig, config_to_dict, run_experiment
from ui_controls import render_hyperparameters

st.set_page_config(page_title="Salinity ML Lab", layout="wide", initial_sidebar_state="expanded")

if "algo_params_store" not in st.session_state:
    st.session_state.algo_params_store = {}


def _save_algo_params(algorithm: str, params: dict) -> None:
    st.session_state.algo_params_store[algorithm] = params


def _load_algo_params(algorithm: str) -> dict:
    return st.session_state.algo_params_store.get(algorithm) or default_params(algorithm)


st.title("Salinity classification lab")
st.caption("Customize every algorithm, pick features, and tune training for small imbalanced datasets.")

# ── Sidebar: data & run settings ──────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    use_upload = st.checkbox("Upload Excel file", value=False)
    data_path = None

    if use_upload:
        uploaded = st.file_uploader("Data.xlsx", type=["xlsx", "xls"])
        if uploaded is not None:
            cache_dir = PROJECT_ROOT / "data" / "uploads"
            cache_dir.mkdir(parents=True, exist_ok=True)
            data_path = str(cache_dir / uploaded.name)
            Path(data_path).write_bytes(uploaded.getvalue())
    else:
        try:
            default_path = str(resolve_data_path())
        except FileNotFoundError:
            default_path = ""
        data_path = st.text_input("Data file path", value=default_path)

    salinity_threshold = st.slider(
        "Ece threshold (≥ → saline class 1)",
        0.0,
        10.0,
        float(DEFAULT_SALINITY_THRESHOLD),
        0.05,
    )

    st.header("Algorithm")
    algo_labels = list_algorithms()
    algorithm = st.selectbox(
        "Model",
        options=list(algo_labels.keys()),
        format_func=lambda k: algo_labels[k],
    )
    compare_all = st.checkbox("Compare all algorithms", value=False)

    st.header("Training")
    sampling = st.selectbox(
        "Resampling",
        options=list(SAMPLING_CHOICES.keys()),
        format_func=lambda k: SAMPLING_CHOICES[k],
    )
    smote_k = st.number_input(
        "SMOTE / ADASYN neighbors (0 = auto)",
        min_value=0,
        max_value=10,
        value=0,
        help="Auto uses min(5, minority_count - 1).",
    )
    normalize = st.checkbox("Standardize features", value=True)

    evaluation_mode = st.selectbox(
        "Evaluation",
        options=["holdout", "stratified_cv"],
        format_func=lambda x: {
            "holdout": "Repeated random hold-out",
            "stratified_cv": "Stratified k-fold (per run)",
        }[x],
    )
    cv_folds = st.slider("CV folds", 2, 10, 5)
    n_runs = st.number_input("Runs", 1, 200, DEFAULT_RUNS)
    test_size = st.slider("Hold-out validation %", 0.1, 0.4, float(DEFAULT_TEST_SIZE), 0.05)

    st.subheader("Decision threshold")
    auto_tune_threshold = st.checkbox(
        "Auto-tune on validation set",
        value=False,
        help="Scans probability thresholds each fold to maximize recall or F1.",
    )
    tune_metric = st.selectbox("Optimize metric", ["recall", "f1"], disabled=not auto_tune_threshold)
    decision_threshold = st.slider(
        "Fixed P(saline) threshold",
        0.05,
        0.95,
        float(DEFAULT_DECISION_THRESHOLD),
        0.05,
        disabled=auto_tune_threshold,
    )

    random_seed = st.number_input("Random seed (0 = random each run)", 0, 10_000, 0)

tab_data, tab_features, tab_params, tab_run, tab_help = st.tabs(
    ["Data", "Features", "Hyperparameters", "Run", "Guide"]
)

feature_mode = "exclude"
drop_columns: list[str] = []
use_columns: list[str] | None = None
model_params = _load_algo_params(algorithm)

# ── Data tab ────────────────────────────────────────────────────────────────
with tab_data:
    if not data_path:
        st.warning("Set a data path in the sidebar or upload a file.")
    else:
        try:
            raw = load_raw_dataframe(data_path)
            st.dataframe(raw.head(25), use_container_width=True)
            st.write(f"**Rows:** {len(raw)} · **Columns:** {len(raw.columns)}")

            y_preview = build_labels(raw, salinity_threshold)
            balance = class_distribution(y_preview.dropna())
            cols = st.columns(max(len(balance), 1))
            for idx, (label, info) in enumerate(balance.items()):
                name = "Saline (1)" if label == 1 else "Non-saline (0)"
                cols[idx].metric(name, f"{info['count']} ({info['percent']}%)")

            if balance.get(1, {}).get("count", 0) < 5:
                st.info("Few saline samples — tune threshold, use recall, and many repeated runs.")
        except Exception as exc:
            st.error(str(exc))

# ── Features tab ──────────────────────────────────────────────────────────
with tab_features:
    if not data_path:
        st.warning("Load data first.")
    else:
        raw = load_raw_dataframe(data_path)
        selectable = get_selectable_feature_columns(raw)

        feature_mode = st.radio(
            "Feature selection mode",
            options=["exclude", "include"],
            format_func=lambda m: (
                "Use all allowed columns except those dropped"
                if m == "exclude"
                else "Use only the columns I select"
            ),
            horizontal=True,
        )

        default_exclude = [c for c in ["Longitude", "Latitude", "Year"] if c in selectable]
        spectral_default = [c for c in selectable if c not in default_exclude]

        if feature_mode == "exclude":
            drop_columns = st.multiselect(
                "Drop these columns",
                options=selectable,
                default=default_exclude,
                key="feat_drop",
            )
            use_columns = None
            st.caption(f"Training with **{len(selectable) - len(drop_columns)}** features.")
        else:
            if "feat_include" not in st.session_state:
                st.session_state["feat_include"] = spectral_default
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("All features", key="btn_all_feat"):
                    st.session_state["feat_include"] = selectable
                    st.rerun()
            with col_b:
                if st.button("Spectral only", key="btn_spectral"):
                    skip = {"Soil_class", "LU_class", "Longitude", "Latitude", "Year"}
                    st.session_state["feat_include"] = [c for c in selectable if c not in skip]
                    st.rerun()
            use_columns = st.multiselect(
                "Use only these columns",
                options=selectable,
                default=st.session_state["feat_include"],
                key="feat_include",
            )
            drop_columns = []
            st.caption(f"Training with **{len(use_columns)}** selected features.")

        try:
            X, y, _ = prepare_dataset(
                data_path,
                salinity_threshold=salinity_threshold,
                drop_columns=drop_columns if feature_mode == "exclude" else None,
                use_columns=use_columns if feature_mode == "include" else None,
            )
            st.success(f"Ready: {len(y)} rows, {X.shape[1]} features")
            st.write(list(X.columns))
        except Exception as exc:
            st.error(str(exc))

# ── Hyperparameters tab ─────────────────────────────────────────────────────
with tab_params:
    if compare_all:
        st.info("Compare-all mode uses **default** hyperparameters for each algorithm.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset to defaults"):
                st.session_state.algo_params_store.pop(algorithm, None)
                st.rerun()
        with c2:
            uploaded_params = st.file_uploader(
                "Load params JSON", type=["json"], key="params_upload"
            )
            if uploaded_params is not None:
                loaded = json.loads(uploaded_params.getvalue())
                st.session_state.algo_params_store[algorithm] = loaded
                st.rerun()

        model_params = render_hyperparameters(algorithm, key_prefix="hp_")
        _save_algo_params(algorithm, model_params)

        st.download_button(
            "Save hyperparameters (JSON)",
            data=json.dumps(model_params, indent=2),
            file_name=f"{algorithm}_params.json",
            mime="application/json",
        )

# ── Run tab ─────────────────────────────────────────────────────────────────
with tab_run:
    run_btn = st.button("Run experiment", type="primary", use_container_width=True)

    if run_btn:
        if not data_path:
            st.error("No data path.")
            st.stop()

        if feature_mode == "include" and not use_columns:
            st.error("Select at least one feature column.")
            st.stop()

        seed = None if random_seed == 0 else int(random_seed)
        smote_k_neighbors = None if smote_k == 0 else int(smote_k)
        algorithms = list(algo_labels.keys()) if compare_all else [algorithm]
        comparison_rows = []

        for algo in algorithms:
            params = default_params(algo) if compare_all else _load_algo_params(algo)

            config = RunConfig(
                data_path=data_path,
                salinity_threshold=salinity_threshold,
                decision_threshold=decision_threshold,
                drop_columns=drop_columns if feature_mode == "exclude" else [],
                use_columns=use_columns if feature_mode == "include" else None,
                algorithm=algo,
                model_params=params,
                sampling=sampling,
                smote_k_neighbors=smote_k_neighbors,
                normalize=normalize,
                n_runs=3 if compare_all else int(n_runs),
                test_size=float(test_size),
                random_seed=seed,
                evaluation_mode=evaluation_mode,
                cv_folds=int(cv_folds),
                auto_tune_threshold=auto_tune_threshold,
                tune_metric=tune_metric,
            )

            progress = st.progress(0.0)
            status = st.empty()

            def on_progress(current, total, label=algo):
                progress.progress(current / total)
                status.text(f"{label}: {current}/{total}")

            try:
                with st.spinner(f"Running {algo}…"):
                    result = run_experiment(config, progress_callback=on_progress)
            except Exception as exc:
                st.error(f"{algo}: {exc}")
                continue
            finally:
                progress.empty()
                status.empty()

            if compare_all:
                row = {
                    "algorithm": algo,
                    "recall_mean": result.summary.get("recall_mean"),
                    "f1_mean": result.summary.get("f1_mean"),
                    "balanced_accuracy_mean": result.summary.get("balanced_accuracy_mean"),
                    "roc_auc_mean": result.summary.get("roc_auc_mean"),
                }
                comparison_rows.append(row)
                continue

            st.success(f"Finished — {algo}")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Recall", f"{result.summary.get('recall_mean', 0):.3f}")
            m2.metric("Precision", f"{result.summary.get('precision_mean', 0):.3f}")
            m3.metric("F1", f"{result.summary.get('f1_mean', 0):.3f}")
            m4.metric("Bal. accuracy", f"{result.summary.get('balanced_accuracy_mean', 0):.3f}")
            m5.metric("ROC-AUC", f"{result.summary.get('roc_auc_mean', 0):.3f}" or "—")

            if result.summary.get("tuned_threshold_mean") is not None:
                st.write(
                    f"Auto-tuned threshold: "
                    f"{result.summary['tuned_threshold_mean']:.2f} "
                    f"± {result.summary.get('tuned_threshold_std', 0):.2f}"
                )

            st.subheader("Summary")
            st.dataframe(
                pd.DataFrame([result.summary]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )

            st.subheader("Per-run / per-fold metrics")
            st.dataframe(result.summary_dataframe(), use_container_width=True)

            if result.feature_importance:
                st.subheader("Feature importance (average)")
                imp_df = (
                    pd.DataFrame(
                        list(result.feature_importance.items()),
                        columns=["feature", "importance"],
                    )
                    .sort_values("importance", ascending=False)
                )
                st.bar_chart(imp_df.set_index("feature"))

            st.download_button(
                "Download full results (JSON)",
                data=json.dumps(
                    {
                        "config": config_to_dict(config),
                        "summary": result.summary,
                        "class_balance": result.class_balance,
                        "features": result.feature_names,
                        "feature_importance": result.feature_importance,
                    },
                    indent=2,
                    default=str,
                ),
                file_name="salinity_run.json",
                mime="application/json",
            )

        if compare_all and comparison_rows:
            st.subheader("Comparison (by recall)")
            st.dataframe(
                pd.DataFrame(comparison_rows).sort_values(
                    "recall_mean", ascending=False, na_position="last"
                ),
                use_container_width=True,
            )


with tab_help:
    st.markdown(
        """
## What this tool does (research workflow)
This app trains a **binary classifier** for salinity using your spreadsheet features and a label derived from:

- **Target column**: `Ece mean`
- **Label rule**: class **1 (saline)** if \(Ece\_mean \ge\) **Ece threshold**, else class **0**

Because the dataset can be **highly imbalanced** (very few saline samples), you should focus on:
- **Recall (saline)**: how many saline profiles you detect
- **Balanced accuracy**: accuracy averaged across classes
- **F1**: balance of precision/recall when you care about false alarms too

## How to use the UI
### 1) Data tab
- Confirm the file loads and columns look correct.
- Adjust the **Ece threshold** and watch how the **class balance** changes.

### 2) Features tab (choose which columns are used)
You have two modes:
- **Exclude mode**: use all allowed columns except the ones you drop.
- **Include mode**: use **only** the selected columns (best when you want full control).

Notes:
- ID columns and leakage columns (like `Ece mean`) are automatically blocked from training.
- “Spectral only” is a fast way to avoid geographic memorization (longitude/latitude/year) and categorical soil/LU columns.

### 3) Hyperparameters tab (full algorithm control)
- Modify every hyperparameter exposed for the chosen algorithm.
- **Save hyperparameters (JSON)** to create a reproducible “model preset”.
- Load a saved JSON later (or share it with another researcher).

### 4) Run tab (evaluation + export)
Evaluation options:
- **Repeated random hold-out**: repeatedly split train/validation and average metrics.
- **Stratified k-fold CV**: splits data into k folds with similar class ratios; better when positives are rare.

Decision threshold options:
- **Fixed threshold**: predicts saline when \(P(saline)\ge\) threshold (default 0.5).
- **Auto-tune threshold**: searches probability cutoffs on each validation fold to maximize **recall** or **F1**.

Exports:
- Download a JSON containing the full config, summary metrics, selected features, and feature importance (when available).

## Training controls (what they mean)
### Resampling (SMOTE / ADASYN)
Used **only on the training split** to reduce class imbalance.

- **none**: no oversampling; rely on class weighting (if the algorithm supports it).
- **SMOTE**: creates synthetic minority samples by interpolating neighbors.
  - **neighbors (k)**: larger k makes smoother synthetic samples; too large can create unrealistic points when the minority class is tiny.
- **ADASYN**: like SMOTE but generates more synthetic samples where the minority class is harder to learn (near boundaries).
  - can help, but may also amplify noise when data is very small.

### Standardize features
Normalization (z-score scaling) is important for distance-based / margin-based models:
- **SVM** and **KNN** usually benefit strongly.
- Tree models (Random Forest / Extra Trees / Gradient Boosting) are less sensitive.

## Algorithms: how they work, when to use them, and hyperparameters
Below, “positive class” means **saline (1)**.

### 1) Random Forest (`random_forest`)
**How it works**: trains many decision trees on bootstrapped samples; averages votes. Handles non-linear feature interactions well.

**When to use**: strong default on tabular data, robust, gives feature importance.

**Hyperparameters**
- **n_estimators (trees)**: more trees → more stable, slower.
- **max_depth**: limits tree depth. Lower values reduce overfitting on small datasets.
- **min_samples_split**: minimum samples to split a node. Higher → smoother/less complex trees.
- **min_samples_leaf**: minimum samples in a leaf. Higher → less variance, often better generalization.
- **max_features**:
  - `sqrt` (common): decorrelates trees, usually good default.
  - `log2`: similar, slightly fewer features per split.
  - `all`: strongest splits, but trees become more similar → can overfit.
- **class_weight**:
  - `balanced`: increases penalty for misclassifying the rare class; often improves recall.
  - `none`: can increase accuracy but often collapses recall when positives are rare.
- **bootstrap**: if enabled, each tree sees a bootstrap sample → more diversity.

### 2) Extra Trees (`extra_trees`)
**How it works**: like Random Forest but splits are more randomized; often faster and sometimes generalizes better.

**When to use**: strong alternative to Random Forest; can be more stable with very small data.

**Hyperparameters**: same meaning as Random Forest.

### 3) Gradient Boosting (`gradient_boosting`)
**How it works**: builds trees sequentially; each new tree corrects errors of the previous ensemble.

**When to use**: can be very accurate, but is easier to overfit on tiny or noisy data.

**Hyperparameters**
- **n_estimators**: more stages → more capacity; combine with smaller learning rate.
- **learning_rate**: shrink contribution of each tree.
  - smaller → usually needs more estimators, but often generalizes better.
- **max_depth**: depth of each weak learner tree.
- **min_samples_split / min_samples_leaf**: regularization, same concept as RF.
- **subsample**: < 1.0 enables stochastic boosting (adds regularization).

### 4) Logistic Regression (`logistic_regression`)
**How it works**: linear decision boundary in feature space; outputs probabilities via sigmoid.

**When to use**: baseline, interpretable, good when signal is mostly linear; works well with scaling.

**Hyperparameters**
- **C**: inverse of regularization strength.
  - smaller C → stronger regularization → simpler model.
- **penalty**:
  - `l2`: smooth shrinkage (default).
  - `l1`: sparse coefficients (feature selection effect).
  - `elasticnet`: mix of l1/l2 (requires solver `saga`).
- **solver**: optimization method; some solvers support only some penalties.
- **max_iter**: increase if convergence warnings occur.
- **class_weight**: `balanced` usually improves minority recall.

### 5) Support Vector Machine (`svm_rbf`)
**How it works**: finds a separating boundary with maximum margin; RBF kernel creates a non-linear boundary.

**When to use**: small datasets where scaling is possible; can work well with few positives but needs careful tuning.

**Hyperparameters**
- **C**: trade-off between margin size and training errors.
  - higher C fits training data more tightly (risk overfitting).
  - lower C is smoother (may improve generalization).
- **kernel**:
  - `rbf`: non-linear (default).
  - `linear`: linear boundary (faster).
  - `poly`: polynomial boundary (rarely needed here).
- **gamma**:
  - `scale` (recommended): adapts to feature variance.
  - `auto`: uses 1 / n_features; sometimes too aggressive/too weak.
- **class_weight**: `balanced` often required to get any recall on rare positives.

### 6) K-Nearest Neighbors (`knn`)
**How it works**: predicts based on the labels of the closest points in feature space.

**When to use**: quick baseline; can work if classes cluster well; sensitive to scaling and noise.

**Hyperparameters**
- **n_neighbors (k)**:
  - small k → sensitive, can overfit.
  - larger k → smoother, may miss rare positives.
- **weights**:
  - `uniform`: each neighbor equally weighted.
  - `distance`: closer points matter more; often better.
- **p**:
  - 1 = Manhattan distance; 2 = Euclidean distance (default).

### 7) Gaussian Naive Bayes (`naive_bayes`)
**How it works**: assumes each feature is conditionally independent given the class and normally distributed.

**When to use**: very small datasets; quick baseline; can perform surprisingly well but assumptions may be violated.

**Hyperparameters**
- **var_smoothing**: adds small variance to stabilize numerical behavior.
  - larger smoothing can prevent overconfidence but can underfit.

## Recommended defaults (for your current class imbalance)
If saline samples are extremely rare:
1. Use **Stratified k-fold CV** (3–5 folds) and multiple runs.
2. Enable **Auto-tune threshold → recall**.
3. Start with **Random Forest** or **Extra Trees**, `class_weight=balanced`, and shallow `max_depth` (3–8).
4. Use SMOTE carefully: keep **neighbors (k)** small when minority class count is tiny.

## CLI usage (same pipeline, no UI)
```bash
python Salinity_2.py --runs 30 --algorithm random_forest
```

Tip: for research reproducibility, keep a fixed `--seed`, and save the UI hyperparameters as JSON.
        """
    )
