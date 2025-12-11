from functools import partial
from typing import Callable, Literal

import numpy as np
from scipy.special import digamma, softmax


def topk(arr: np.ndarray, k: int):
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices


# Estimating LLM Uncertainy with Evidence
# arxiv:2502.00290v5
# https://github.com/MaHuanAAA/logtoku/blob/main/SenU/metrics.py
def eu(logits: np.ndarray, top_k: int = 2) -> float:
    if len(logits) < top_k:
        raise ValueError("Logits array length is less than top_k.")
    top_values, _ = topk(logits, top_k)
    mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
    return mean_scores


def au(logits: np.ndarray, top_k: int = 2) -> float:
    if len(logits) < top_k:
        raise ValueError("Logits array length is less than top_k.")
    top_values = np.partition(logits, -top_k)[-top_k:]
    alpha = np.array([top_values])
    alpha_0 = alpha.sum(axis=1, keepdims=True)
    psi_alpha_k_plus_1 = digamma(alpha + 1)
    psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
    result = -(alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
    return result.sum(axis=1)[0]


def prob(logits: np.ndarray, top_k: int = 1) -> float:
    logits = softmax(logits)
    if len(logits) < top_k:
        raise ValueError("Logits array length is less than top_k.")
    top_values, _ = topk(logits, top_k)
    return top_values[0]


def entropy(logits: np.ndarray) -> float:
    probs = softmax(logits)
    entropy_value = -np.sum(probs * np.log(probs + 1e-10))
    return entropy_value


def get_eu(
    mode: Literal["eu", "prob", "entropy", "au", "eu_2", "au_2"] = "prob",
    k: int | None = None,
) -> Callable[[np.ndarray], float]:
    match mode:
        case "eu":
            if k is None:
                raise ValueError("k must be provided for 'eu' mode.")
            return partial(eu, top_k=k)
        case "eu_2":
            return partial(eu, top_k=2)
        case "prob":
            return partial(prob, top_k=1)
        case "entropy":
            return entropy
        case "au":
            if k is None:
                raise ValueError("k must be provided for 'au' mode.")
            return partial(au, top_k=k)
        case "au_2":
            return partial(au, top_k=2)
        case _:
            raise ValueError(f"Unsupported mode: {mode}")
