import os
import sys
import time
import json
import ctypes
import numpy as np
import faiss
import matplotlib.pyplot as plt

# PYTHONPATH fallback and PithosMIDB Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB


def generate_hypersphere_vectors(n: int, dim: int) -> np.ndarray:
    raw = np.random.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return raw / norms


def get_pithos_tiers(dim: int) -> np.ndarray:
    """Returns Matryoshka cascading tiers to enable early-exit pruning."""
    if dim <= 64:
        return np.array([dim], dtype=np.int32)
    elif dim == 128:
        return np.array([64, 128], dtype=np.int32)
    elif dim == 256:
        return np.array([64, 128, 256], dtype=np.int32)
    elif dim == 384:
        return np.array([64, 128, 256, 384], dtype=np.int32)
    else:
        return np.array([64, 128, 256, 512, dim], dtype=np.int32)


def cleanup_index_files(db_file: str):
    for ext in ["", "_ids.bin", "_metadata.bin"]:
        p = db_file + ext
        if os.path.exists(p):
            os.remove(p)
    k = 0
    while True:
        p = f"{db_file}_tier_{k}.bin"
        if os.path.exists(p):
            os.remove(p)
            k += 1
        else:
            break


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------

def run_pithos_search(
    db_vectors: np.ndarray, queries: np.ndarray,
    dim: int, k: int,
) -> float:
    """Compile, load, warmup, time a Pithos batch search. Returns seconds."""
    actual_dim = max(dim, 64)
    tiers = get_pithos_tiers(actual_dim)

    if dim < 64:
        db_vectors = np.pad(db_vectors, ((0, 0), (0, 64 - dim)), mode="constant")
        queries = np.pad(queries, ((0, 0), (0, 64 - dim)), mode="constant")

    engine = PithosMIDB()
    db_file = f"pithos_sweep_dim_{dim}"
    ids = np.arange(db_vectors.shape[0], dtype=np.int64)

    status = engine.compile_index_file(db_file, 1, 1737400, actual_dim, tiers, ids, db_vectors)
    if status != 0:
        engine.close()
        raise RuntimeError(f"Index compile failed for dim {dim}")

    # Orthogonal weight matrix for clean SVD energy computation
    q_mat, _ = np.linalg.qr(np.random.normal(size=(actual_dim, actual_dim)))
    mock_weights = q_mat.astype(np.float32)
    status = engine.load_index("sweep", db_file, mock_weights, actual_dim)
    if status != 0:
        engine.close()
        raise RuntimeError(f"Index load failed for dim {dim}")

    # Set optimal chunk size to fully utilize multi-core on smaller datasets (100k)
    engine.set_chunk_size("sweep", 5000)

    # Warmup pass
    warmup_q = queries[:min(5, queries.shape[0])]
    engine.batch_search("sweep", warmup_q, k)

    # Timed pass
    t0 = time.perf_counter()
    engine.batch_search("sweep", queries, k)
    elapsed = time.perf_counter() - t0

    engine.close()
    cleanup_index_files(db_file)
    return elapsed


def run_faiss_search(
    db_vectors: np.ndarray, queries: np.ndarray, dim: int, k: int,
) -> float:
    """Build FAISS Flat L2, warmup, time a batch search. Returns seconds."""
    index = faiss.IndexFlatL2(dim)
    index.add(db_vectors)
    # Warmup
    index.search(queries[:min(5, queries.shape[0])], k)
    t0 = time.perf_counter()
    index.search(queries, k)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DIMENSIONS = [16, 32, 64, 128, 256, 384, 512, 768, 1024]
NUM_RECORDS = 100_000
K = 100
NUM_QUERIES_SINGLE = 1
NUM_QUERIES_MULTI = 100


def main():
    print("=" * 72)
    print("    PITHOS vs FAISS -- Dimensionality Crossover Sweep")
    print("=" * 72)
    np.random.seed(42)

    sweep_results = []

    for dim in DIMENSIONS:
        print(f"\n--- D = {dim} ---", flush=True)
        db_vectors = generate_hypersphere_vectors(NUM_RECORDS, dim)

        # ---- Run A: Single-Query Paradigm (pure latency) ----
        q_single = generate_hypersphere_vectors(NUM_QUERIES_SINGLE, dim)

        print(f"  [A] Single-Query (N=1) ...", end=" ", flush=True)
        t_pithos_single = run_pithos_search(db_vectors, q_single, dim, K)
        t_faiss_single = run_faiss_search(db_vectors, q_single, dim, K)

        pithos_lat_us = t_pithos_single * 1e6
        faiss_lat_us = t_faiss_single * 1e6
        winner_single = "pithos" if pithos_lat_us < faiss_lat_us else "faiss"
        print(
            f"Pithos {pithos_lat_us:,.1f} us | FAISS {faiss_lat_us:,.1f} us "
            f"-> {winner_single}",
            flush=True,
        )

        # ---- Run B: Multi-Query Paradigm (throughput) ----
        q_multi = generate_hypersphere_vectors(NUM_QUERIES_MULTI, dim)

        print(f"  [B] Multi-Query  (N={NUM_QUERIES_MULTI}) ...", end=" ", flush=True)
        t_pithos_multi = run_pithos_search(db_vectors, q_multi, dim, K)
        t_faiss_multi = run_faiss_search(db_vectors, q_multi, dim, K)

        total_comparisons = NUM_RECORDS * NUM_QUERIES_MULTI
        pithos_mvps = (total_comparisons / t_pithos_multi) / 1e6
        faiss_mvps = (total_comparisons / t_faiss_multi) / 1e6
        winner_multi = "pithos" if pithos_mvps > faiss_mvps else "faiss"
        print(
            f"Pithos {pithos_mvps:,.2f} MVPS | FAISS {faiss_mvps:,.2f} MVPS "
            f"-> {winner_multi}",
            flush=True,
        )

        sweep_results.append({
            "dim": dim,
            "single_query": {
                "pithos_latency_us": round(pithos_lat_us, 2),
                "faiss_latency_us": round(faiss_lat_us, 2),
                "winner": winner_single,
            },
            "multi_query": {
                "pithos_mvps": round(pithos_mvps, 2),
                "faiss_mvps": round(faiss_mvps, 2),
                "winner": winner_multi,
            },
        })

    # ------------------------------------------------------------------
    # Export telemetry
    # ------------------------------------------------------------------
    metrics = {"dimensionality_sweep": sweep_results}
    os.makedirs("temp/benchmark_data", exist_ok=True)
    metrics_path = "temp/benchmark_data/pithos_sweep_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics exported to {metrics_path}")

    # ------------------------------------------------------------------
    # Consolidated Crossover Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("    CROSSOVER ANALYSIS REPORT")
    print("=" * 72)

    header = (
        f"{'D':>6}  |  {'Single-Query Latency (us)':^36}  |  {'Multi-Query Throughput (MVPS)':^38}"
    )
    sub = (
        f"{'':>6}  |  {'Pithos':>10}  {'FAISS':>10}  {'Winner':>10}  "
        f"|  {'Pithos':>10}  {'FAISS':>10}  {'Speedup':>10}"
    )
    print(header)
    print(sub)
    print("-" * len(sub))

    for r in sweep_results:
        sq = r["single_query"]
        mq = r["multi_query"]
        if mq["pithos_mvps"] > mq["faiss_mvps"]:
            speedup = f"{mq['pithos_mvps'] / mq['faiss_mvps']:.1f}x"
        else:
            speedup = f"-{mq['faiss_mvps'] / mq['pithos_mvps']:.1f}x"
        print(
            f"{r['dim']:>6}  |  {sq['pithos_latency_us']:>10.1f}  "
            f"{sq['faiss_latency_us']:>10.1f}  {sq['winner']:>10}  "
            f"|  {mq['pithos_mvps']:>10.2f}  {mq['faiss_mvps']:>10.2f}  "
            f"{speedup:>10}"
        )

    # Identify crossover boundaries
    print("\n" + "-" * 72)
    crossover_dims_single = []
    crossover_dims_multi = []
    for i in range(1, len(sweep_results)):
        prev, curr = sweep_results[i - 1], sweep_results[i]
        if prev["single_query"]["winner"] != curr["single_query"]["winner"]:
            crossover_dims_single.append(
                f"D={prev['dim']}->{curr['dim']} ({prev['single_query']['winner']} -> {curr['single_query']['winner']})"
            )
        if prev["multi_query"]["winner"] != curr["multi_query"]["winner"]:
            crossover_dims_multi.append(
                f"D={prev['dim']}->{curr['dim']} ({prev['multi_query']['winner']} -> {curr['multi_query']['winner']})"
            )

    if crossover_dims_single:
        for c in crossover_dims_single:
            print(f"  Single-Query Crossover: {c}")
    else:
        dominant = sweep_results[0]["single_query"]["winner"]
        print(f"  Single-Query: No crossover detected. {dominant} dominates all dimensions.")

    if crossover_dims_multi:
        for c in crossover_dims_multi:
            print(f"  Multi-Query  Crossover: {c}")
    else:
        dominant = sweep_results[0]["multi_query"]["winner"]
        print(f"  Multi-Query:  No crossover detected. {dominant} dominates all dimensions.")

    print("=" * 72)

    # ------------------------------------------------------------------
    # Plotting: dual-panel crossover chart
    # ------------------------------------------------------------------
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    fig, (ax_lat, ax_tp) = plt.subplots(1, 2, figsize=(14, 6))

    dims = [r["dim"] for r in sweep_results]
    p_lat = [r["single_query"]["pithos_latency_us"] for r in sweep_results]
    f_lat = [r["single_query"]["faiss_latency_us"] for r in sweep_results]
    p_mvps = [r["multi_query"]["pithos_mvps"] for r in sweep_results]
    f_mvps = [r["multi_query"]["faiss_mvps"] for r in sweep_results]

    # Left panel: Single-Query Latency
    ax_lat.plot(dims, p_lat, marker="o", linewidth=2.5, color="#2e7d32", label="Pithos")
    ax_lat.plot(dims, f_lat, marker="s", linewidth=2.5, color="#1565c0", label="FAISS Flat L2")
    ax_lat.set_title("Single-Query Latency (lower is better)", fontsize=12, fontweight="bold")
    ax_lat.set_xlabel("Vector Dimensionality (D)", fontsize=11, fontweight="bold")
    ax_lat.set_ylabel("Latency (us)", fontsize=11, fontweight="bold")
    ax_lat.set_xscale("log", base=2)
    ax_lat.set_yscale("log")
    ax_lat.set_xticks(dims)
    ax_lat.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_lat.legend(loc="upper left", frameon=True, fontsize=9)
    ax_lat.grid(True, which="both", linestyle="--", alpha=0.5)

    # Right panel: Multi-Query Throughput
    ax_tp.plot(dims, p_mvps, marker="o", linewidth=2.5, color="#2e7d32", label="Pithos")
    ax_tp.plot(dims, f_mvps, marker="s", linewidth=2.5, color="#1565c0", label="FAISS Flat L2")
    ax_tp.set_title("Multi-Query Throughput (higher is better)", fontsize=12, fontweight="bold")
    ax_tp.set_xlabel("Vector Dimensionality (D)", fontsize=11, fontweight="bold")
    ax_tp.set_ylabel("Throughput (MVPS)", fontsize=11, fontweight="bold")
    ax_tp.set_xscale("log", base=2)
    ax_tp.set_xticks(dims)
    ax_tp.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_tp.legend(loc="upper right", frameon=True, fontsize=9)
    ax_tp.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Pithos vs FAISS Flat L2 -- Crossover Curve (100k records, K=100)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/crossover_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig("assets/crossover_curve.svg", bbox_inches="tight")
    plt.close()
    print("\nCrossover plot saved to assets/crossover_curve.png")

    # Auto-update README.md with live metrics
    try:
        from generate_graphics import update_readme
        update_readme()
    except Exception as e:
        print(f"Warning: Could not update README.md ({e})")


if __name__ == "__main__":
    main()
