import os
import time
import numpy as np
import faiss
import json

def main():
    print("Loading actual lunar vectors and queries...")
    if not os.path.exists("db_vectors.npy") or not os.path.exists("queries.npy"):
        print("[Error] db_vectors.npy or queries.npy not found. Run ingest_pipeline.py and query_generator.py first.")
        return
        
    db_vectors = np.load("db_vectors.npy") # shape (1000000, 384)
    queries = np.load("queries.npy")       # shape (278, 384)
    
    print(f"Database vectors shape: {db_vectors.shape}")
    print(f"Queries shape: {queries.shape}")
    
    # 1. FAISS Flat Index L2
    print("Running FAISS IndexFlatL2 baseline...")
    faiss_index = faiss.IndexFlatL2(384)
    faiss_index.add(db_vectors)
    
    # Warm up search
    _ = faiss_index.search(queries[:10], 100)
    
    t0 = time.perf_counter()
    _, _ = faiss_index.search(queries, 100)
    t_faiss = time.perf_counter() - t0
    faiss_mvps = (db_vectors.shape[0] * queries.shape[0]) / t_faiss / 1e6
    print(f"FAISS Flat Index L2 search time: {t_faiss*1000:.2f} ms ({faiss_mvps:.2f} MVPS)")
    
    # 2. Sequential JIT / Single-threaded Loop Simulation
    print("Running Sequential JIT compiled loop simulation...")
    # Single-threaded L2 distance loop in python over a subset
    subset_size = 1000
    subset = db_vectors[:subset_size]
    t0 = time.perf_counter()
    for q in queries[:10]:
        diff = subset - q
        dist = np.sum(diff * diff, axis=1)
        _ = np.argsort(dist)[:100]
    t_seq_raw = time.perf_counter() - t0
    
    # Scale search time to 1,000,000 records and 278 queries
    t_seq_scaled = (t_seq_raw / (10 * subset_size)) * (queries.shape[0] * db_vectors.shape[0])
    seq_mvps = (db_vectors.shape[0] * queries.shape[0]) / t_seq_scaled / 1e6
    print(f"Sequential JIT baseline simulated search time: {t_seq_scaled*1000:.2f} ms ({seq_mvps:.2f} MVPS)")
    
    # Save baseline metrics
    with open("baselines_metrics.json", "w") as f:
        json.dump({
            "faiss_mvps": faiss_mvps,
            "sequential_mvps": seq_mvps
        }, f, indent=2)
    print("Metrics saved to baselines_metrics.json")

if __name__ == "__main__":
    main()
