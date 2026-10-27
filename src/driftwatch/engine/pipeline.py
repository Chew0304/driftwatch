import numpy as np
from typing import Dict, Any, List, Optional
from driftwatch.sketches.bloom_filter import BloomFilter
from driftwatch.sketches.count_min_sketch import CountMinSketch
from driftwatch.sketches.hyperloglog import HyperLogLog
from driftwatch.drift.detector import VectorDriftDetector


class StreamingDriftPipeline:
    """
    Unified streaming telemetry and drift detection pipeline.
    Orchestrates sublinear sketches and spatial vector drift monitoring
    over continuous micro-batches of telemetry events.
    """

    def __init__(
        self,
        baseline_embeddings: np.ndarray,
        expected_unique_clients: int = 50_000,
        bloom_fp_rate: float = 0.01,
        cms_epsilon: float = 0.005,
        cms_delta: float = 0.01,
        hll_precision: int = 10,
        drift_std_multiplier: float = 2.0,
    ):
        self.bloom_filter = BloomFilter(
            expected_elements=expected_unique_clients,
            false_positive_rate=bloom_fp_rate,
        )
        self.cms = CountMinSketch(epsilon=cms_epsilon, delta=cms_delta)
        self.hll = HyperLogLog(precision=hll_precision)
        self.drift_detector = VectorDriftDetector(
            baseline_embeddings=baseline_embeddings,
            threshold_std_multiplier=drift_std_multiplier,
        )

        self.total_events_processed = 0
        self.duplicate_events_dropped = 0

    def ingest_event(
        self,
        event_id: str,
        client_id: str,
        tokens: List[str],
    ) -> bool:
        """
        Process a single streaming metadata event.
        Returns False if the event was identified as a duplicate by the Bloom Filter.
        """
        if event_id in self.bloom_filter:
            self.duplicate_events_dropped += 1
            return False

        self.bloom_filter.add(event_id)
        self.hll.add(client_id)
        for token in tokens:
            self.cms.update(token)

        self.total_events_processed += 1
        return True

    def ingest_batch(
        self,
        event_ids: List[str],
        client_ids: List[str],
        token_lists: List[List[str]],
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process a streaming batch of events and evaluate vector distribution drift.
        """
        accepted = 0
        for ev_id, c_id, toks in zip(event_ids, client_ids, token_lists):
            if self.ingest_event(ev_id, c_id, toks):
                accepted += 1

        drift_report = None
        if embeddings is not None and len(embeddings) > 0:
            drift_report = self.drift_detector.evaluate_batch(embeddings)

        return {
            "batch_size": len(event_ids),
            "accepted_events": accepted,
            "dropped_duplicates": len(event_ids) - accepted,
            "estimated_unique_clients": self.hll.count(),
            "drift_report": drift_report,
        }

    def get_token_frequency(self, token: str) -> int:
        """Query bounded frequency estimate for a token."""
        return self.cms.estimate(token)
