import gc
import sys
import time
import tracemalloc

sys.path.insert(0, "src")

from driftwatch.sketches.bloom_filter import BloomFilter
from driftwatch.sketches.count_min_sketch import CountMinSketch
from driftwatch.sketches.hyperloglog import HyperLogLog


def format_bytes(num_bytes: float) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / 1024:.2f} KB"


def run_benchmark():
    print("=" * 80)
    print(" DRIFTWATCH: EMPIRICAL RESOURCE & THROUGHPUT BENCHMARK ")
    print("=" * 80)

    n_items = 150_000
    dataset = [f"telemetry_metric_uuid_{i}" for i in range(n_items)]

    # 1. Cardinality: Naive Set vs. HyperLogLog
    print(f"\n[Scenario 1] Cardinality Estimation over {n_items:,} Distinct Elements")
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    naive_set = set(dataset)
    t_set = time.perf_counter() - t0
    _, peak_set = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    hll = HyperLogLog(precision=11)
    for item in dataset:
        hll.add(item)
    t_hll = time.perf_counter() - t0
    _, peak_hll = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    est_card = hll.count()
    error_pct = abs(est_card - n_items) / n_items * 100

    print(f"  • Native set:      Memory = {format_bytes(peak_set):<10} | Time = {t_set:.3f}s | Throughput = {n_items / t_set:,.0f} ops/s")
    print(f"  • HyperLogLog:     Memory = {format_bytes(peak_hll):<10} | Time = {t_hll:.3f}s | Throughput = {n_items / t_hll:,.0f} ops/s")
    print(f"    --> RAM Savings: {(1 - (peak_hll / peak_set)) * 100:.2f}% | Estimated: {est_card:,} (Error: {error_pct:.2f}%)")

    # 2. Membership: Naive Set vs. BloomFilter
    print(f"\n[Scenario 2] Set Membership Filtering over {n_items:,} Elements (Target p=0.01)")
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    membership_set = set(dataset)
    t_mset = time.perf_counter() - t0
    _, peak_mset = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    bf = BloomFilter(expected_elements=n_items, false_positive_rate=0.01)
    for item in dataset:
        bf.add(item)
    t_bf = time.perf_counter() - t0
    _, peak_bf = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  • Native set:      Memory = {format_bytes(peak_mset):<10} | Time = {t_mset:.3f}s | Throughput = {n_items / t_mset:,.0f} ops/s")
    print(f"  • BloomFilter:     Memory = {format_bytes(peak_bf):<10} | Time = {t_bf:.3f}s | Throughput = {n_items / t_bf:,.0f} ops/s")
    print(f"    --> RAM Savings: {(1 - (peak_bf / peak_mset)) * 100:.2f}% (Allocated bits: {bf.m:,})")

    # 3. Frequency: Naive Dict vs. Count-Min Sketch
    print(f"\n[Scenario 3] Frequency Estimation over {n_items * 2:,} Stream Events")
    stream_events = dataset + dataset[:50_000] + dataset[:50_000]

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    naive_dict = {}
    for item in stream_events:
        naive_dict[item] = naive_dict.get(item, 0) + 1
    t_dict = time.perf_counter() - t0
    _, peak_dict = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    cms = CountMinSketch(epsilon=0.005, delta=0.01)
    for item in stream_events:
        cms.update(item)
    t_cms = time.perf_counter() - t0
    _, peak_cms = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  • Native dict:     Memory = {format_bytes(peak_dict):<10} | Time = {t_dict:.3f}s | Throughput = {len(stream_events) / t_dict:,.0f} ops/s")
    print(f"  • CountMinSketch:  Memory = {format_bytes(peak_cms):<10} | Time = {t_cms:.3f}s | Throughput = {len(stream_events) / t_cms:,.0f} ops/s")
    print(f"    --> RAM Savings: {(1 - (peak_cms / peak_dict)) * 100:.2f}% (Table: {cms.depth}x{cms.width} counters)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmark()
