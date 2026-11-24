"""Brute-force exact k-nearest-neighbor search — the ground truth every index is measured against.

Approach: a max-heap of size k. Walk every distance once; keep only the k best seen so far.
  Time:   O(n·d) to compute distances + O(n log k) for the heap
  Memory: O(k) for the heap — it never holds more than k items, so the same pattern
          works on a stream of vectors that never fits in memory at once.

Alternatives to benchmark later (write the numbers in docs/journal.md):
  1. Full sort of all distances            — O(n log n)
  2. heapify all n, then pop k             — O(n + k log n), but O(n) memory
  3. `np.argpartition` then sort the top k — O(n + k log k), runs in C
"""

import heapq

import numpy as np

from vektor.distance import cosine_distance, l2_squared

_METRICS = {"l2": l2_squared, "cosine": cosine_distance}


def search(query: np.ndarray, vectors: np.ndarray, k: int, metric: str = "l2") -> np.ndarray:
    """Return the indices of the `k` vectors closest to `query`, closest first.

    query:   shape (d,)
    vectors: shape (n, d)
    metric:  "l2" (squared Euclidean) or "cosine"
    returns: int array of shape (k,)

    Raises ValueError if k < 1, k > n, or the metric is unknown.
    Ties are broken by index: the vector stored first wins.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {sorted(_METRICS)}")
    n = len(vectors)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")

    distances = _METRICS[metric](query, vectors)

    # Python's heapq is a MIN-heap. Storing (-distance, -index) turns it into a max-heap,
    # so heap[0] is always the WORST of the k best: largest distance, and on a tie, the
    # larger index (so earlier vectors win ties).
    heap: list[tuple[float, int]] = []
    for i, d in enumerate(distances):
        if len(heap) < k:
            heapq.heappush(heap, (-d, -i))
        elif d < -heap[0][0]:  # strictly better than the current worst -> replace it
            heapq.heapreplace(heap, (-d, -i))

    # Undo the negation and order closest-first (distance, then index).
    best = sorted((-neg_d, -neg_i) for neg_d, neg_i in heap)
    return np.array([i for _, i in best], dtype=np.int64)
