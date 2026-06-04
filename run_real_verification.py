import os
import sys
import time
import numpy as np
from benchmark import LcvkEngine

DB_FILE = "lunar_real_data.bin"

def hamming_distance_matrix(queries, db_vectors):
    # queries: (Q, 6), db_vectors: (N, 6)
    # Reinterpret as uint64 to avoid sign issues during bitwise operations
    q_u64 = queries.view(np.uint64)
    db_u64 = db_vectors.view(np.uint64)
    
    # Broadcast XOR and count bits
    dists = np.zeros((q_u64.shape[0], db_u64.shape[0]), dtype=np.int32)
    for q_idx in range(q_u64.shape[0]):
        xor_result = q_u64[q_idx, :, np.newaxis] ^ db_u64.T # shape (6, N)
        # Vectorized popcount for each 64-bit word
        popcounts = np.zeros(db_u64.shape[0], dtype=np.int32)
        for i in range(6):
            word_xor = np.ascontiguousarray(xor_result[i])
            # Use NumPy vectorization: convert to bytes to count bits fast
            bytes_arr = word_xor.view(np.uint8).reshape(-1, 8)
            # Count set bits in each byte using a lookup table
            bit_counts = np.array([bin(b).count("1") for b in range(256)], dtype=np.int32)
            popcounts += np.sum(bit_counts[bytes_arr], axis=1)
        dists[q_idx] = popcounts
    return dists

def main():
    # 1. Load queries, labels, and database for analysis
    print("Loading queries, labels, and database vectors...")
    if not os.path.exists("queries.npy") or not os.path.exists("db_labels.npy"):
        print("[Error] Files missing. Please run ingest_pipeline.py and query_generator.py.")
        sys.exit(1)
        
    queries = np.load("queries.npy")       # shape (278, 6)
    db_labels = np.load("db_labels.npy")   # shape (1000000,)
    
    # Read first 10,000 records from DB file for distance analysis
    with open(DB_FILE, "rb") as f:
        f.seek(64) # skip header
        records_raw = f.read(10000 * 64)
    records_arr = np.frombuffer(records_raw, dtype=np.uint8).reshape(-1, 64)
    db_vectors = records_arr[:, 8:56].copy().view(np.int64) # shape (10000, 6)
    db_labels_subset = db_labels[:10000]
    
    # Compute pairwise Hamming distances for the first 50 queries against 10000 DB records
    print("Analyzing Hamming distance distribution...")
    dists = hamming_distance_matrix(queries[:50], db_vectors) # shape (50, 10000)
    
    is_seven = (db_labels_subset == 7)
    
    dists_to_sevens = dists[:, is_seven]
    dists_to_others = dists[:, ~is_seven]
    
    mean_sevens = np.mean(dists_to_sevens)
    std_sevens = np.std(dists_to_sevens)
    min_sevens = np.min(dists_to_sevens)
    max_sevens = np.max(dists_to_sevens)
    
    mean_others = np.mean(dists_to_others)
    std_others = np.std(dists_to_others)
    min_others = np.min(dists_to_others)
    max_others = np.max(dists_to_others)
    
    print("\n" + "="*80)
    print("                 LCVK HAMMING DISTANCE DISTRIBUTION REPORT              ")
    print("========================================================================")
    print(f" Query Digit vs Target Digit (7):")
    print(f"  - Mean Distance           : {mean_sevens:.2f} bits")
    print(f"  - Std Dev                 : {std_sevens:.2f} bits")
    print(f"  - Range (Min / Max)       : {min_sevens} / {max_sevens} bits")
    print("------------------------------------------------------------------------")
    print(f" Query Digit vs Other Digits (0-6, 8-9):")
    print(f"  - Mean Distance           : {mean_others:.2f} bits")
    print(f"  - Std Dev                 : {std_others:.2f} bits")
    print(f"  - Range (Min / Max)       : {min_others} / {max_others} bits")
    print("========================================================================\n")
    
    # Find the optimal threshold
    # We want a threshold T that is:
    # - Higher than most of target sevens' distances (high recall)
    # - Lower than almost all other digits' distances (high precision)
    # Let's search for T that maximizes F1-Score on our subset
    best_f1 = 0
    best_threshold = 0
    for T in range(int(min_sevens), int(max_others)):
        is_resonant_subset = (np.sum(dists <= T, axis=0) >= 7) # popcount >= 7
        tp = np.sum(is_resonant_subset & is_seven)
        fp = np.sum(is_resonant_subset & ~is_seven)
        fn = np.sum(~is_resonant_subset & is_seven)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = T
            
    print(f"Suggested optimal threshold: {best_threshold} (F1-score on subset: {best_f1*100.0:.2f}%)")
    
    # 2. Run the actual native verification with the suggested threshold
    print(f"\nRe-running native verification with threshold = {best_threshold}...")
    
    import platform
    if platform.system() == "Darwin":
        so_paths = [
            "./target/lunar_core.dylib",
            "./build-output/liblunar_core.dylib",
            "./liblunar_core.dylib",
            "./target/lunar_core.so",
            "./build-output/liblunar_core.so",
        ]
    else:
        so_paths = [
            "./build-output/liblunar_core.so",
            "./liblunar_core.so",
            "./target/lunar_core.so",
        ]
    
    lib_path = None
    for p in so_paths:
        if os.path.exists(p):
            lib_path = p
            break
            
    if not lib_path:
        print("[Error] LCVK native library not found.")
        sys.exit(1)
        
    engine = LcvkEngine(lib_path)
    status = engine.load_index("lunar_real", DB_FILE)
    if status != 0:
        print(f"[Error] Failed to load index. Code: {status}")
        sys.exit(1)
        
    total_records = engine.size("lunar_real")
    voting_mask = np.zeros(total_records, dtype=np.uint8)
    
    # Update thresholds to the suggested value
    families = np.load("families.npy")
    thresholds = np.full(queries.shape[0], best_threshold, dtype=np.int32)
    
    t_vote_start = time.perf_counter()
    resonant_count = engine.query_planetary_grid("lunar_real", queries, families, thresholds, voting_mask)
    t_vote_sec = time.perf_counter() - t_vote_start
    t_vote_ms = t_vote_sec * 1000.0
    
    throughput_mvps = (total_records * queries.shape[0]) / t_vote_sec / 1e6
    print(f"Scan completed in {t_vote_ms:.3f} ms ({throughput_mvps:.2f} MVPS).")
    
    # Compute full dataset metrics
    bit_counts = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    mask_popcounts = bit_counts[voting_mask]
    
    is_resonant = (mask_popcounts >= 7)
    is_seven = (db_labels == 7)
    
    tp = np.sum(is_resonant & is_seven)
    fp = np.sum(is_resonant & ~is_seven)
    fn = np.sum(~is_resonant & is_seven)
    tn = np.sum(~is_resonant & ~is_seven)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*80)
    print("                 LCVK REAL-DATA CLASSIFICATION METRICS                  ")
    print("========================================================================")
    print(f" Target Digit               : 7 (Lunar Cave Entrance Anchor)")
    print(f" Total Database Records     : {total_records:,}")
    print(f" Actual Target Count ('7's) : {np.sum(is_seven):,}")
    print(f" Resonant Matches (Found)   : {resonant_count:,}")
    print("------------------------------------------------------------------------")
    print(f" Confusion Matrix")
    print(f"  - True Positives (TP)     : {tp:,}")
    print(f"  - False Positives (FP)    : {fp:,}")
    print(f"  - False Negatives (FN)    : {fn:,}")
    print(f"  - True Negatives (TN)     : {tn:,}")
    print("------------------------------------------------------------------------")
    print(f" Performance Metrics")
    print(f"  - Precision               : {precision * 100.0:6.2f}%")
    print(f"  - Recall                  : {recall * 100.0:6.2f}%")
    print(f"  - F1-Score                : {f1 * 100.0:6.2f}%")
    print("========================================================================\n")
    
    engine.close()
    
    # Clean up files
    for f in [DB_FILE, "queries.npy", "families.npy", "thresholds.npy", "db_labels.npy", "raw_sevens.npy"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()
