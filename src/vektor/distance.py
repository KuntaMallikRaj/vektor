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
    # (a - b) for every vector -> square each difference -> add up each row
    return ((vectors - query) ** 2).sum(axis=1)


def cosine_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine_similarity, from `query` to every row of `vectors`.

    Same direction -> 0, perpendicular -> 1, opposite -> 2.
    Only direction matters, not length: [1, 1] and [5, 5] have distance 0.
    An all-zero vector has no direction; it gets distance 1 (unrelated) instead of NaN.
    """
    # cos(angle) = (a · b) / (|a| * |b|)  — dot product divided by both lengths
    dots = vectors @ query
    lengths = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    lengths = np.where(lengths == 0, 1.0, lengths)  # avoid dividing by zero
    return 1.0 - dots / lengths
