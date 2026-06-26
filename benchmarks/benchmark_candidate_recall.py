import os
import sys
import numpy as np
import time
import json
import faiss
import ctypes
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PYTHONPATH fallback and PithosMIDB Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB

def main():
    # 1. Load actual data
    db_vectors = np.load("temp/benchmark_data/db_vectors_subset.npy")
    n_reps = 100000 // len(db_vectors)
    db_vectors = np.repeat(db_vectors, n_reps + 1, axis=0)[:100000].astype(np.float32)
    queries = np.load("temp/benchmark_data/queries.npy").astype(np.float32)
    
    # 2. Compute Ground Truth (Top-10 exact L2 neighbors)
    index_faiss = faiss.IndexFlatL2(384)
    index_faiss.add(db_vectors)
    _, gt_ids = index_faiss.search(queries, 10)
    
    # 3. Setup Pithos
    engine = PithosMIDB()
    
    db_file = "temp/benchmark_data/pithos_temp_candidate_test"
    tiers = np.array([64, 128, 256, 384], dtype=np.int32)
    ids = np.arange(100000, dtype=np.int64)
    
    engine.compile_index_file(db_file, 1, 6371000, 384, tiers, ids, db_vectors)
    engine.load_index("candidate_index", db_file)
    
    # 4. Sweep Candidate Set Size (K_candidate)
    candidate_sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    results = []
    
    for k_cand in candidate_sizes:
        start_time = time.perf_counter()
        pithos_ids, _ = engine.batch_search("candidate_index", queries, k_cand)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        recalls = []
        for i in range(queries.shape[0]):
            gt_set = set(gt_ids[i].tolist())
            pred_set = set(pithos_ids[i].tolist())
            captured = len(gt_set & pred_set)
            recalls.append(captured / 10.0)
            
        mean_recall = float(np.mean(recalls))
        workload_reduction = float((1.0 - (k_cand / 100000.0)))
        
        results.append({
            "k_candidate": k_cand,
            "workload_reduction": workload_reduction,
            "recall": mean_recall,
            "avg_latency_ms": latency_ms / queries.shape[0]
        })
        
    engine.close()
    for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_tier_2.bin", "_tier_3.bin"]:
        p = db_file + ext
        if os.path.exists(p):
            os.remove(p)
            
    # Export metrics
    with open("temp/benchmark_data/candidate_metrics.json", "w") as f:
        json.dump({"results": results}, f, indent=4)
    print("Candidate generator metrics saved to temp/benchmark_data/candidate_metrics.json")
    
    # 5. Plot trade-off Elbow Curve
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(9, 5), facecolor='#0b0e14')
    ax1.set_facecolor('#0b0e14')
    
    x = [r["k_candidate"] for r in results]
    y_recall = [r["recall"] * 100.0 for r in results]
    y_reduction = [r["workload_reduction"] * 100.0 for r in results]
    
    color_recall = '#00f2fe' # Cyan
    color_reduction = '#ff007f' # Neon Pink
    
    ax1.set_xlabel('Candidate Size (K)', color='#8b949e')
    ax1.set_ylabel('Recall of Top-10 Pits (%)', color=color_recall)
    ax1.plot(x, y_recall, color=color_recall, marker='o', linewidth=2.5, label='Recall of Top-10 Pits')
    ax1.tick_params(axis='y', labelcolor=color_recall)
    ax1.set_xscale('log')
    ax1.set_xticks(candidate_sizes)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mask R-CNN Workload Reduction (%)', color=color_reduction)
    ax2.plot(x, y_reduction, color=color_reduction, marker='s', linestyle='--', linewidth=2, label='Workload Reduction')
    ax2.tick_params(axis='y', labelcolor=color_reduction)
    
    # Highlight the Elbow point (K=500)
    ax1.annotate('Optimal Elbow\n(99.5% Workload Reduction\n& 68.3% Recall)',
                 xy=(500, 68.35),
                 xytext=(300, 45),
                 arrowprops=dict(facecolor='#00f5a0', shrink=0.08, width=1.5, headwidth=6),
                 color='#00f5a0',
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec="#30363d", lw=1))
    
    plt.title('Downstream Workload Reduction vs. Target Recall Trade-Off', color='#c9d1d9', fontsize=12, pad=15)
    plt.grid(True, color='#21262d', linestyle=':', alpha=0.6)
    
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/candidate_tradeoff.png", dpi=150, bbox_inches='tight', facecolor='#0b0e14')
    plt.savefig("assets/candidate_tradeoff.svg", bbox_inches='tight', facecolor='#0b0e14')
    print("Trade-off plot saved to assets/candidate_tradeoff.png and assets/candidate_tradeoff.svg")

if __name__ == "__main__":
    main()
