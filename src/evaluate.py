import pandas as pd
import numpy as np


def precision_k(y_true: pd.DataFrame, y_score: pd.DataFrame, k: int) -> float:
    results = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "score": np.asarray(y_score),
    })

    top_k = results.sample(frac=1, random_state=42).nlargest(k, "score")
    return top_k["y_true"].eq("yes").mean()


def score_precision_10(estimator, X, y):
    yes_idx = list(estimator.classes_).index("yes")
    scores = estimator.predict_proba(X)[:, yes_idx]
    k = max(1, int(len(y) * 0.1))
    return precision_k(y, scores, k=k)
