import numpy as np
import pytest

from vektor.brute_force import search


def test_points_on_a_line_closest_first():
    vectors = np.array([[10.0], [1.0], [5.0], [2.0]])
    result = search(np.array([0.0]), vectors, k=3)
    assert list(result) == [1, 3, 2]  # values 1, 2, 5


def test_k_equals_n_returns_everything_sorted():
    vectors = np.array([[3.0], [1.0], [2.0]])
    assert list(search(np.array([0.0]), vectors, k=3)) == [1, 2, 0]


def test_matches_full_sort_on_random_data():
    rng = np.random.default_rng(42)
    vectors = rng.random((1000, 32))
    query = rng.random(32)
    expected = np.argsort(((vectors - query) ** 2).sum(axis=1))[:10]
    np.testing.assert_array_equal(search(query, vectors, k=10), expected)


def test_cosine_metric():
    vectors = np.array([[0.0, 1.0], [10.0, 0.1], [-1.0, 0.0]])
    # [10, 0.1] is far in L2 but points almost exactly the same way as the query
    assert search(np.array([1.0, 0.0]), vectors, k=1, metric="cosine")[0] == 1


@pytest.mark.parametrize("k", [0, -1, 5])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        search(np.zeros(2), np.zeros((4, 2)), k=k)


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        search(np.zeros(2), np.zeros((4, 2)), k=1, metric="manhattan")
