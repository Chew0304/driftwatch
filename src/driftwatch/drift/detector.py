import numpy as np
from driftwatch.drift.spatial_index import KDTree


class VectorDriftDetector:
    """
    Evaluates distribution drift in streaming embedding batches
    relative to a reference baseline distribution.
    """

    def __init__(self, baseline_embeddings: np.ndarray, threshold_std_multiplier: float = 2.0):
        self.baseline_embeddings = np.asarray(baseline_embeddings, dtype=np.float64)
        if len(self.baseline_embeddings) < 10:
            raise ValueError("Baseline embeddings must contain at least 10 samples.")

        # Split baseline: 80% to build KDTree reference index, 20% held out for calibration
        n_samples = len(self.baseline_embeddings)
        split_idx = int(n_samples * 0.8)
        self.ref_embeddings = self.baseline_embeddings[:split_idx]
        self.calib_embeddings = self.baseline_embeddings[split_idx:]

        self.tree = KDTree(self.ref_embeddings)
        self.multiplier = threshold_std_multiplier

        # Calibration points are genuine held-out samples from the baseline distribution
        baseline_dists = [self.tree.query_nearest(v)[0] for v in self.calib_embeddings]
        self.baseline_mean_dist = float(np.mean(baseline_dists))
        self.baseline_std_dist = float(np.std(baseline_dists))
        self.drift_threshold = self.baseline_mean_dist + (self.multiplier * self.baseline_std_dist)

    def evaluate_batch(self, stream_embeddings: np.ndarray) -> dict:
        """
        Evaluate a batch of streaming embedding vectors.
        Triggers an alert when batch mean distance exceeds the empirical threshold.
        """
        stream_embeddings = np.asarray(stream_embeddings, dtype=np.float64)
        distances = [self.tree.query_nearest(v)[0] for v in stream_embeddings]
        batch_mean_dist = float(np.mean(distances))

        return {
            "batch_mean_distance": batch_mean_dist,
            "baseline_mean_distance": self.baseline_mean_dist,
            "threshold": self.drift_threshold,
            "drift_detected": bool(batch_mean_dist > self.drift_threshold),
        }
