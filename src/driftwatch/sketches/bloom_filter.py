import hashlib
import math

try:
    import mmh3
    HAS_MMH3 = True
except ImportError:
    HAS_MMH3 = False


class BloomFilter:
    """
    Space-efficient probabilistic set membership testing.
    Guarantees:
      - Zero false negatives: If check(x) is False, x was definitely never added.
      - Tunable false positive rate p for a specified capacity n.
      - Space complexity: O(m) bits, independent of key lengths.
    """

    def __init__(self, expected_elements: int = 100_000, false_positive_rate: float = 0.01):
        if expected_elements <= 0:
            raise ValueError("expected_elements must be greater than 0")
        if not (0 < false_positive_rate < 1):
            raise ValueError("false_positive_rate must be strictly between 0 and 1")

        self.n = expected_elements
        self.p = false_positive_rate

        # Optimal bit array size m and hash function count k
        self.m = math.ceil(-(self.n * math.log(self.p)) / (math.log(2) ** 2))
        self.k = math.ceil((self.m / self.n) * math.log(2))

        # Bit array using bytearray (8 bits per byte)
        self.byte_count = math.ceil(self.m / 8)
        self.bit_array = bytearray(self.byte_count)
        self.count = 0

    def _hashes(self, item: str):
        """Yield k bit positions for the given item using seeded hashing."""
        key = str(item)
        for seed in range(self.k):
            if HAS_MMH3:
                pos = mmh3.hash(key, seed=seed, signed=False) % self.m
            else:
                digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
                pos = int(digest, 16) % self.m
            yield pos

    def add(self, item: str) -> None:
        """Insert an item by setting its k hashed bit positions to 1."""
        for pos in self._hashes(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self.bit_array[byte_idx] |= (1 << bit_idx)
        self.count += 1

    def contains(self, item: str) -> bool:
        """Check membership. Returns False (definitely not present) or True (likely present)."""
        for pos in self._hashes(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def __contains__(self, item: str) -> bool:
        return self.contains(item)
