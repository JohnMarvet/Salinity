"""Optional hyperparameter tuning (Random Forest)."""

from sklearn.model_selection import GridSearchCV

from sampling import apply_sampling


def tune_model(model, X_train, y_train, sampling="smote", random_state=42):
    X_res, y_res = apply_sampling(X_train, y_train, method=sampling, random_state=random_state)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 6, 10],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    grid.fit(X_res, y_res)
    return grid.best_estimator_
