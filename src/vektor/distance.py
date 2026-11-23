"""Distance functions — how Vektor measures "closeness" between vectors.

YOUR TASK (Week 1): implement both functions. Make `tests/test_distance.py` pass.
Rule: no Python `for` loops over the vectors — use NumPy so it's fast.
"""

import numpy as np


def l2_squared(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from `query` to every row of `vectors`.

    query:   shape (d,)
    vectors: shape (n, d)
    returns: shape (n,) — smaller means closer.

    Why *squared*? Taking the square root never changes which vector is closest,
    so we skip it and save work. (Think about why that's true.)
    """
    raise NotImplementedError


def cosine_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine_similarity, from `query` to every row of `vectors`.

    Same direction -> 0, perpendicular -> 1, opposite -> 2.
    Only direction matters, not length: [1, 1] and [5, 5] have distance 0.
    """
    raise NotImplementedError
