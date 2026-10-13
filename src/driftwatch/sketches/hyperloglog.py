import hashlib
import math
import numpy as np

try:
    import mmh3
    HAS_MMH3 = True
except ImportError:
    HAS_MMH3 = False


class HyperLogLog:
    """
    Sublinear-space distinct element counter (cardinality estimation).
    Guarantees:
      - Relative error approximately 1.04 / sqrt(m) where m = 2^p.
      - Memory footprint: O(2^p) bytes, strictly constant regardless of stream length.
    """

    def __init__(self, precision: int = 10):
        if not (4 <= precision <= 16):
            raise ValueError("Precision p must be between 4 and 16.")

        self.p = precision
        self.m = 1 << precision  # 2^p registers
        self.registers = np.zeros(self.m, dtype=np.uint8)

        # Precompute Flajolet alpha correction factor
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def _hash(self, item: str) -> int:
        """Produce a uniform 64-bit unsigned integer hash."""
        key = str(item).encode("utf-8")
        if HAS_MMH3:
            return mmh3.hash64(key, seed=0, signed=False)[0]
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], byteorder="big")

    def _rho(self, w: int, max_bits: int) -> int:
        """Count leading zeros plus one in binary representation."""
        if w == 0:
            return max_bits + 1
        leading_zeros = max_bits - w.bit_length()
        return leading_zeros + 1

    def add(self, item: str) -> None:
        """Update register based on hashed prefix and leading zeros."""
        x = self._hash(item)
        # First p bits determine register index
        j = x >> (64 - self.p)
        # Remaining (64 - p) bits determine leading zero run
        w = x & ((1 << (64 - self.p)) - 1)
        r = self._rho(w, 64 - self.p)
        if r > self.registers[j]:
            self.registers[j] = r

    def count(self) -> int:
        """Compute estimated cardinality using normalized harmonic mean."""
        indicator = np.sum(2.0 ** (-self.registers.astype(np.float64)))
        raw_estimate = (self.alpha * (self.m ** 2)) / indicator

        # Small range correction (Linear Counting for sparse registers)
        if raw_estimate <= 2.5 * self.m:
            zeros = np.count_nonzero(self.registers == 0)
            if zeros != 0:
                return int(round(self.m * math.log(self.m / zeros)))

        return int(round(raw_estimate))
