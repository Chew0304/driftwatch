from driftwatch.sketches.count_min_sketch import CountMinSketch
from driftwatch.sketches.bloom_filter import BloomFilter


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

    # Fundamental guarantee: zero false negatives
    for key in inserted_keys:
        assert key in bf, f"False negative encountered on inserted key: {key}"


def test_bloom_filter_false_positive_rate():
    expected_n = 2000
    target_p = 0.05
    bf = BloomFilter(expected_elements=expected_n, false_positive_rate=target_p)

    # Insert expected_n items
    for i in range(expected_n):
        bf.add(f"user_session_{i}")

    # Query unseen items and evaluate empirical false positive rate
    test_queries = 2000
    false_positives = 0
    for i in range(test_queries):
        unseen_key = f"unseen_random_session_{i}"
        if unseen_key in bf:
            false_positives += 1

    empirical_p = false_positives / test_queries
    # Verify empirical rate stays close to theoretical bound
    assert empirical_p <= target_p + 0.02, f"Empirical FP rate {empirical_p} exceeded target {target_p}"
