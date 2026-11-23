import numpy as np
import pytest

from vektor.distance import cosine_distance, l2_squared


def test_l2_known_values():
    query = np.array([0.0, 0.0])
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(l2_squared(query, vectors), [25.0, 1.0, 0.0])


def test_l2_matches_naive_loop():
    rng = np.random.default_rng(0)
    query = rng.random(16)
    vectors = rng.random((100, 16))
    expected = [sum((q - v) ** 2 for q, v in zip(query, row)) for row in vectors]
    np.testing.assert_allclose(l2_squared(query, vectors), expected)


def test_l2_returns_one_distance_per_vector():
    assert l2_squared(np.zeros(8), np.ones((50, 8))).shape == (50,)


def test_cosine_direction_cases():
    query = np.array([1.0, 0.0])
    vectors = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    # same direction, perpendicular, opposite
    np.testing.assert_allclose(cosine_distance(query, vectors), [0.0, 1.0, 2.0], atol=1e-9)


@pytest.mark.parametrize("scale", [0.5, 1.0, 10.0])
def test_cosine_ignores_length(scale):
    rng = np.random.default_rng(1)
    query = rng.random(8)
    vectors = rng.random((20, 8))
    np.testing.assert_allclose(
        cosine_distance(query, vectors), cosine_distance(query * scale, vectors * scale), atol=1e-9
    )
