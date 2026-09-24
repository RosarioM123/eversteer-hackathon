"""
Model registry: maps a config's `model` string to a factory function.

These are generic, standard estimators used as stand-ins. Once the real
feature set / size / types are known, this registry can be extended without
touching the runner or branch files - that's the point of registering by
name rather than hardcoding model construction everywhere.

No model here has been tuned to any NYC-specific data, because none exists
yet in this codebase.
"""

from typing import Callable, Dict, Any


def _build_ridge(params: Dict[str, Any]):
    from sklearn.linear_model import Ridge
    return Ridge(**{"alpha": 1.0, **params})


def _build_lasso(params: Dict[str, Any]):
    from sklearn.linear_model import Lasso
    return Lasso(**{"alpha": 0.001, **params})


def _build_shallow_gbm(params: Dict[str, Any]):
    from sklearn.ensemble import GradientBoostingRegressor
    defaults = {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05}
    return GradientBoostingRegressor(**{**defaults, **params})


def _build_full_gbm(params: Dict[str, Any]):
    from sklearn.ensemble import GradientBoostingRegressor
    defaults = {"max_depth": 5, "n_estimators": 500, "learning_rate": 0.02}
    return GradientBoostingRegressor(**{**defaults, **params})


def _build_random_forest(params: Dict[str, Any]):
    from sklearn.ensemble import RandomForestRegressor
    defaults = {"n_estimators": 300, "max_depth": None}
    return RandomForestRegressor(**{**defaults, **params})


def _build_neutral_baseline(params: Dict[str, Any]):
    """Predicts the training-fold mean target for every row. This is B0."""
    class NeutralBaseline:
        def __init__(self):
            self._mean = 0.0

        def fit(self, X, y):
            self._mean = sum(y) / len(y) if len(y) else 0.0
            return self

        def predict(self, X):
            return [self._mean] * len(X)

    return NeutralBaseline()


MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "neutral_baseline": _build_neutral_baseline,   # B0
    "ridge": _build_ridge,                          # B2 candidate
    "lasso": _build_lasso,
    "shallow_gbm": _build_shallow_gbm,              # nonlinearity probe
    "full_gbm": _build_full_gbm,                    # B4 candidate
    "random_forest": _build_random_forest,          # B5 / diversity candidate
}


def build_model(model_key: str, params: Dict[str, Any]):
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. Registered models: "
            f"{list(MODEL_REGISTRY.keys())}. Add a new factory in "
            f"models/registry.py before referencing it in an ExperimentConfig."
        )
    return MODEL_REGISTRY[model_key](params)
