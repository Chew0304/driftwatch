from driftwatch.sketches.count_min_sketch import CountMinSketch
from driftwatch.sketches.bloom_filter import BloomFilter
from driftwatch.sketches.hyperloglog import HyperLogLog


def test_count_min_sketch_accuracy():
    cms = CountMinSketch(epsilon=0.01, delta=0.01)

    for _ in range(500):
        cms.update("heavy_query_token")

    for i in range(100):
        cms.update(f"noise_{i}")

    estimate = cms.estimate("heavy_query_token")
    assert estimate >= 500
    assert estimate <= 500 + (cms.epsilon * cms.total_count)


def test_bloom_filter_no_false_negatives():
    bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
    inserted_keys = [f"request_id_{i}" for i in range(500)]

    for key in inserted_keys:
        bf.add(key)

    for key in inserted_keys:
        assert key in bf, f"False negative on key: {key}"


def test_bloom_filter_false_positive_rate():
    expected_n = 2000
    target_p = 0.05
    bf = BloomFilter(expected_elements=expected_n, false_positive_rate=target_p)

    for i in range(expected_n):
        bf.add(f"user_session_{i}")

    test_queries = 2000
    false_positives = 0
    for i in range(test_queries):
        unseen_key = f"unseen_random_session_{i}"
        if unseen_key in bf:
            false_positives += 1

    empirical_p = false_positives / test_queries
    assert empirical_p <= target_p + 0.02


def test_hyperloglog_cardinality_estimation():
    # Precision p = 10 -> m = 1024 registers, theoretical std error ~= 3.25%
    hll = HyperLogLog(precision=10)
    actual_distinct = 5000

    for i in range(actual_distinct):
        hll.add(f"unique_client_id_{i}")

    # Add duplicates to ensure idempotency
    for i in range(1500):
        hll.add(f"unique_client_id_{i}")

    estimate = hll.count()

    # Estimate should fall comfortably within a 10% error margin (approx 3 std errors)
    relative_error = abs(estimate - actual_distinct) / actual_distinct
    assert relative_error <= 0.10, f"HLL error too high: {relative_error:.4f} (estimate: {estimate})"
