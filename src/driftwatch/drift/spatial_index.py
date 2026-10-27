import numpy as np


class KDNode:
    """Node representation in a k-dimensional tree."""

    def __init__(self, point: np.ndarray, axis: int, left=None, right=None, data_idx: int = -1):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
        self.data_idx = data_idx


class KDTree:
    """
    Spatial partitioning tree for fast nearest-neighbor search
    in D-dimensional continuous vector spaces.
    Reduces query complexity from brute-force O(N) to average O(log N).
    """

    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError("Data must be a 2D array of shape (N, D).")
        self.n_samples, self.dim = self.data.shape
        self.root = self._build_tree(np.arange(self.n_samples), depth=0)

    def _build_tree(self, indices: np.ndarray, depth: int):
        if len(indices) == 0:
            return None

        axis = depth % self.dim
        # Sort indices along current split dimension
        sorted_indices = indices[np.argsort(self.data[indices, axis])]
        median_idx = len(sorted_indices) // 2

        node_idx = sorted_indices[median_idx]
        return KDNode(
            point=self.data[node_idx],
            axis=axis,
            left=self._build_tree(sorted_indices[:median_idx], depth + 1),
            right=self._build_tree(sorted_indices[median_idx + 1:], depth + 1),
            data_idx=node_idx,
        )

    def query_nearest(self, target: np.ndarray) -> tuple[float, int]:
        """Find the Euclidean distance and index of the nearest neighbor."""
        target = np.asarray(target, dtype=np.float64)
        best_dist = float("inf")
        best_idx = -1

        def _search(node):
            nonlocal best_dist, best_idx
            if node is None:
                return

            dist = np.linalg.norm(target - node.point)
            if dist < best_dist:
                best_dist = dist
                best_idx = node.data_idx

            axis = node.axis
            diff = target[axis] - node.point[axis]

            first = node.left if diff < 0 else node.right
            second = node.right if diff < 0 else node.left

            _search(first)
            # Prune subtree if the hyperplane boundary is farther than current best
            if abs(diff) < best_dist:
                _search(second)

        _search(self.root)
        return float(best_dist), int(best_idx)
