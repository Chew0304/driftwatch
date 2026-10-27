import numpy as np
from driftwatch.engine.pipeline import StreamingDriftPipeline


def test_pipeline_deduplication_and_sketches():
    np.random.seed(42)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=(100, 8))
    pipeline = StreamingDriftPipeline(baseline_embeddings=baseline)

    # Ingest event 1
    assert pipeline.ingest_event("req_001", "user_alpha", ["token_a", "token_b"])
    # Re-ingest same event ID -> Bloom Filter must drop it
    assert not pipeline.ingest_event("req_001", "user_alpha", ["token_a"])
    assert pipeline.duplicate_events_dropped == 1

    # Ingest event 2
    assert pipeline.ingest_event("req_002", "user_beta", ["token_a", "token_c"])

    # Check CMS frequency bounds
    assert pipeline.get_token_frequency("token_a") >= 2
    assert pipeline.get_token_frequency("token_c") >= 1


def test_pipeline_batch_ingestion_with_drift():
    np.random.seed(42)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=(200, 16))
    pipeline = StreamingDriftPipeline(baseline_embeddings=baseline, drift_std_multiplier=2.0)

    # In-distribution batch
    event_ids = [f"evt_{i}" for i in range(50)]
    client_ids = [f"client_{i % 10}" for i in range(50)]
    token_lists = [["inference", "query"] for _ in range(50)]
    normal_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(50, 16))

    result_normal = pipeline.ingest_batch(event_ids, client_ids, token_lists, normal_embeddings)
    assert result_normal["accepted_events"] == 50
    assert not result_normal["drift_report"]["drift_detected"]

    # Shifted/Drifted batch
    shifted_event_ids = [f"evt_shift_{i}" for i in range(50)]
    shifted_client_ids = [f"client_{i % 5}" for i in range(50)]
    shifted_tokens = [["anomalous", "payload"] for _ in range(50)]
    shifted_embeddings = np.random.normal(loc=3.0, scale=1.0, size=(50, 16))

    result_shifted = pipeline.ingest_batch(
        shifted_event_ids, shifted_client_ids, shifted_tokens, shifted_embeddings
    )
    assert result_shifted["drift_report"]["drift_detected"]
