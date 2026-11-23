"""Quality metrics for approximate search.

YOUR TASK (Week 1): implement `recall_at_k`. Make `tests/test_metrics.py` pass.
"""

import numpy as np


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of the true top-k neighbors that the search actually found.

    found: indices returned by the search, shape (k,) for one query or (q, k) for q queries
    truth: true nearest-neighbor indices, same shape (may be longer than k — use the first k)
    returns: recall in [0.0, 1.0]; for many queries, the average over all of them.

    Order does not matter: found=[3, 1], truth=[1, 3] -> 1.0
    """
    raise NotImplementedError
