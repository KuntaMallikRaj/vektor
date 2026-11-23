"""Brute-force exact k-nearest-neighbor search — the ground truth every index is measured against.

YOUR TASK (Week 1): implement `search`. Make `tests/test_brute_force.py` pass.

Try it three ways and time each one on 1M vectors (write the numbers in docs/journal.md):
  1. Full sort of all distances            — O(n log n)
  2. A size-k max-heap with `heapq`        — O(n log k)   <- the DSA way
  3. `np.argpartition` then sort the top k — O(n + k log k)
"""

import numpy as np


def search(query: np.ndarray, vectors: np.ndarray, k: int, metric: str = "l2") -> np.ndarray:
    """Return the indices of the `k` vectors closest to `query`, closest first.

    query:   shape (d,)
    vectors: shape (n, d)
    metric:  "l2" (squared Euclidean) or "cosine" — use the functions in vektor.distance
    returns: int array of shape (k,)

    Raise ValueError if k < 1, k > n, or the metric is unknown.
    """
    raise NotImplementedError
