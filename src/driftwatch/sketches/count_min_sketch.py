import hashlib
import math
import numpy as np

try:
    import mmh3
    HAS_MMH3 = True
except ImportError:
    HAS_MMH3 = False


class CountMinSketch:
    """
    Sublinear-space frequency estimation using the Count-Min Sketch algorithm.
    Guarantees:
      - Point query error: at most epsilon * N with probability 1 - delta.
      - Space complexity: O(w * d) counters.
    """

    def __init__(self, epsilon: float = 0.001, delta: float = 0.01):
        self.epsilon = epsilon
        self.delta = delta

        # w = ceil(e / epsilon), d = ceil(ln(1 / delta))
        self.width = math.ceil(math.e / epsilon)
        self.depth = math.ceil(math.log(1.0 / delta))

        # 2D counter table
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.total_count = 0

    def _hash(self, key: str, i: int) -> int:
        if HAS_MMH3:
            return mmh3.hash(str(key), seed=i) % self.width
        digest = hashlib.sha256(f"{i}:{key}".encode("utf-8")).hexdigest()
        return int(digest, 16) % self.width

    def update(self, key: str, value: int = 1) -> None:
        """Increment count for a given key across all depth rows."""
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.table[i, idx] += value
        self.total_count += value

    def estimate(self, key: str) -> int:
        """Query estimated frequency (minimum across hash rows)."""
        estimates = [self.table[i, self._hash(key, i)] for i in range(self.depth)]
        return min(estimates)
