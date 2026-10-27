import numpy as np
from driftwatch.drift.detector import VectorDriftDetector
from driftwatch.drift.spatial_index import KDTree


def test_kdtree_exact_match():
    np.random.seed(42)
    data = np.random.randn(200, 8)
    tree = KDTree(data)

    # A point existing in the tree should have nearest neighbor distance 0
    query_point = data[25]
    dist, idx = tree.query_nearest(query_point)

    assert np.isclose(dist, 0.0)
    assert idx == 25


def test_kdtree_matches_brute_force():
    np.random.seed(42)
    data = np.random.randn(150, 6)
    tree = KDTree(data)

    test_query = np.random.randn(6)
    kdtree_dist, kdtree_idx = tree.query_nearest(test_query)

    # Brute-force calculation for validation
    brute_dists = np.linalg.norm(data - test_query, axis=1)
    brute_min_dist = np.min(brute_dists)
    brute_min_idx = np.argmin(brute_dists)

    assert np.isclose(kdtree_dist, brute_min_dist)
    assert kdtree_idx == brute_min_idx


def test_drift_detection_normal_vs_shifted():
    np.random.seed(42)
    # Baseline: Gaussian centered at origin
    baseline = np.random.normal(loc=0.0, scale=1.0, size=(500, 16))
    detector = VectorDriftDetector(baseline, threshold_std_multiplier=2.0)

    # In-distribution batch: Should NOT trigger drift
    in_distribution_batch = np.random.normal(loc=0.0, scale=1.0, size=(80, 16))
    normal_result = detector.evaluate_batch(in_distribution_batch)
    assert not normal_result["drift_detected"]

    # Shifted distribution batch: Should trigger drift
    shifted_batch = np.random.normal(loc=2.5, scale=1.0, size=(80, 16))
    drifted_result = detector.evaluate_batch(shifted_batch)
    assert drifted_result["drift_detected"]
