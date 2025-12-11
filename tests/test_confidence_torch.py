import numpy as np
import pytest
import torch

from diagnose_w_confidence.server.confidence import get_eu as get_eu_numpy
from diagnose_w_confidence.server.confidence_torch import get_eu as get_eu_torch


@pytest.mark.parametrize(
    "mode,k",
    [("eu", k) for k in range(1, 10)]
    + [("au", k) for k in range(1, 10)]
    + [
        ("prob", None),
        ("entropy", None),
        ("eu_2", None),
        ("au_2", None),
    ]
    * 3,  # repeat to increase test coverage
)
def test_get_eu_pairwise(mode, k):
    eu_func_numpy = get_eu_numpy(mode=mode, k=k)
    eu_func_torch = get_eu_torch(mode=mode, k=k)

    logits = torch.randn(1000)

    actual = eu_func_numpy(logits.numpy())
    expected = eu_func_torch(logits)

    assert np.isclose(actual, expected), f"Results do not match for mode={mode}, k={k}"
