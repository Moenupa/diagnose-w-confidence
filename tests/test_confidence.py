import numpy as np
import pytest

from diagnose_w_confidence.server.confidence import get_eu
from diagnose_w_confidence.server.confidence_original import get_eu as get_eu_original


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
    eu_func_ours = get_eu(mode=mode, k=k)
    eu_func_original = get_eu_original(mode=mode, k=k)

    logits = np.random.randn(1000).astype(np.float32)

    actual = eu_func_ours(logits)
    expected = eu_func_original(logits)

    assert np.isclose(actual, expected), f"Results do not match for mode={mode}, k={k}"
