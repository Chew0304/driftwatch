from driftwatch.sketches.count_min_sketch import CountMinSketch


def test_count_min_sketch_accuracy():
    # Setup CMS with bounded error parameters
    cms = CountMinSketch(epsilon=0.01, delta=0.01)

    # Insert 500 occurrences of a simulated heavy-hitter token
    for _ in range(500):
        cms.update("heavy_query_token")

    # Insert background noise
    for i in range(100):
        cms.update(f"noise_{i}")

    estimate = cms.estimate("heavy_query_token")

    # CMS never underestimates: estimate >= true_count
    assert estimate >= 500

    # Overestimation error is bounded by epsilon * total_count
    max_allowable_bound = 500 + (cms.epsilon * cms.total_count)
    assert estimate <= max_allowable_bound
