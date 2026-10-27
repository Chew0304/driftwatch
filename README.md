# DriftWatch

[![DriftWatch CI](https://github.com/Chew0304/driftwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Chew0304/driftwatch/actions)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-green)
![Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen)

**DriftWatch** is an engine designed for online ML inference telemetry and real-time semantic drift detection over high-throughput event streams. 

By replacing naive collections with probabilistic sketching structures and accelerating high-dimensional nearest-neighbor searches via spatial partitioning, DriftWatch achieves sublinear heap allocation with bounded theoretical error.

---


## Architecture Overview

DriftWatch processes streaming telemetry through a 4-stage pipeline that eliminates unbounded memory growth:

### Stage 1: Ingestion & Deduplication Gate (`driftwatch.sketches.bloom_filter`)
* **Streaming Ingestion:** Ingests high-velocity telemetry streams containing request IDs, query tokens, and embedding vectors.
* **Replay Filtering:** Evaluates incoming event IDs against a bit-level Bloom Filter to drop replay attacks and duplicate payloads with zero false negatives ($p \le 0.01$).
* **Pipeline Dispatch:** Routes verified, non-duplicate events concurrently to the sketching estimators and the vector drift engine.

### Stage 2: Sublinear Telemetry & Sketching (`driftwatch.sketches`)
* **Frequency Tracking (`CountMinSketch`):** Aggregates token and feature occurrences across a fixed $d \times w$ integer grid, guaranteeing zero underestimation and bounded error ($\le \epsilon \cdot N$).
* **Cardinality Tracking (`HyperLogLog`):** Tracks active client and session counts using 64-bit hashing and register leading-zero extraction, achieving $\approx 1.04/\sqrt{m}$ error in $\approx 1\text{ KB}$ of RAM.

### Stage 3: High-Dimensional Spatial Indexing (`driftwatch.drift.spatial_index`)
* **Spatial Partitioning:** Indexes baseline training embeddings into an orthogonal hyperplane $k$-d tree.
* **Nearest-Neighbor Pruning:** Replaces exhaustive $\mathcal{O}(N)$ distance sweeps with recursive hyperplane branch pruning to query nearest neighbors in $\mathcal{O}(\log N)$ average time.

### Stage 4: Statistical Drift Detection (`driftwatch.drift.detector`)
* **Held-out Calibration:** Uses an 80/20 reference split to calibrate baseline distance statistics ($\mu, \sigma$) and avoid zero-distance self-query threshold collapse.
* **Batch Drift Evaluation:** Measures mean nearest-neighbor distances for incoming embedding batches; triggers automated alerts whenever batch distance exceeds the empirical threshold ($\mu + k\sigma$).


---

## Empirical Benchmark Results

Benchmarked on an AMD Ryzen 7 / Ubuntu WSL2 environment under 150,000–300,000 continuous event streams:

| Workload / Primitive | Baseline (Python Standard) | DriftWatch Implementation | Peak Memory Reduction | Throughput | Theoretical Guarantee |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cardinality Estimation** (150k Unique Items) | `set` (6.00 MB) | **HyperLogLog** (6.09 KB) | **99.90%** | ~282,000 ops/s | $\text{SE} \approx 1.04 / \sqrt{m}$ (Observed: 2.11%) |
| **Membership Filtering** (150k Items, $p=0.01$) | `set` (6.00 MB) | **Bloom Filter** (176.90 KB) | **97.12%** | ~132,000 ops/s | Zero False Negatives; $p \le 0.01$ |
| **Frequency Tracking** (300k Stream Events) | `dict` (5.50 MB) | **Count-Min Sketch** (22.38 KB) | **99.60%** | ~95,000 ops/s | Error bounded by $\le \epsilon \cdot N$ with prob $1 - \delta$ |

---

## Algorithmic Components

### 1. Count-Min Sketch (`driftwatch.sketches.count_min_sketch`)
- Computes bounded $(\epsilon, \delta)$ frequency approximations for high-velocity streaming items.
- Configured with depth $d = \lceil\ln(1/\delta)\rceil$ and width $w = \lceil e/\epsilon\rceil$.
- Guarantees zero underestimation: $\hat{a}_i \ge a_i$, with bounded error $\hat{a}_i \le a_i + \epsilon \|a\|_1$.

### 2. Space-Optimized Bloom Filter (`driftwatch.sketches.bloom_filter`)
- Deduplication gate operating on a compact bit array (`bytearray`).
- Zero false negatives guaranteed: if `contains(x) == False`, $x$ was never processed.
- Parameterized with optimal hash count $k = \frac{m}{n}\ln 2$ and bit array size $m = -\frac{n \ln p}{(\ln 2)^2}$.

### 3. HyperLogLog (`driftwatch.sketches.hyperloglog`)
- Sublinear cardinality estimator using 64-bit hashing and register leading-zero extraction.
- Mitigates outlier distortion using normalized harmonic averaging and Flajolet correction factors.
- Incorporates linear counting for small-range cardinality stabilization.

### 4. KDTree Spatial Index & Drift Detector (`driftwatch.drift`)
- Orthogonal hyperplane spatial partitioning tree reducing nearest-neighbor search from $\mathcal{O}(N)$ brute-force to $\mathcal{O}(\log N)$.
- Employs a held-out calibration split on reference embeddings to avoid self-query threshold collapse.
- Evaluates statistical batch distance deviations to flag semantic distribution shift in streaming embeddings.

---

## Installation & Testing

```bash
# Clone and enter repo
git clone [https://github.com/Chew0304/driftwatch.git](https://github.com/Chew0304/driftwatch.git)
cd driftwatch

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt pytest

# Execute unit test suite
python3 -m pytest -v

# Run empirical benchmark suite
python3 benchmarks/memory_profile.py
