import numpy as np
import pytest

from vektor.metrics import recall_at_k


def test_perfect_recall():
    assert recall_at_k(np.array([1, 2, 3]), np.array([1, 2, 3]), k=3) == 1.0


def test_order_does_not_matter():
    assert recall_at_k(np.array([3, 1]), np.array([1, 3]), k=2) == 1.0


def test_half_recall():
    assert recall_at_k(np.array([1, 9]), np.array([1, 2]), k=2) == 0.5


def test_truth_longer_than_k_uses_first_k():
    # SIFT ground truth stores 100 neighbors per query; recall@10 uses only the first 10
    assert recall_at_k(np.array([5, 6]), np.array([5, 6, 7, 8]), k=2) == 1.0


def test_many_queries_are_averaged():
    found = np.array([[1, 2], [7, 8]])
    truth = np.array([[1, 2], [7, 0]])
    assert recall_at_k(found, truth, k=2) == pytest.approx(0.75)  # (1.0 + 0.5) / 2
