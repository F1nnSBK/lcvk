import os
import sys
import numpy as np

def main():
    # Load raw embeddings of pits saved by ingest_pipeline.py
    if not os.path.exists("temp/benchmark_data/raw_pits.npy"):
        print("[Error] raw_pits.npy not found. Please run ingest_pipeline.py first.")
        sys.exit(1)
        
    raw_pits = np.load("temp/benchmark_data/raw_pits.npy")
    print(f"Loaded {raw_pits.shape[0]} raw embeddings of lunar pits.")
    
    # Select 278 queries
    num_queries = 278
    if raw_pits.shape[0] < num_queries:
        print(f"[Warning] Only found {raw_pits.shape[0]} pits. Repeating to reach {num_queries}.")
        indices = np.arange(num_queries) % raw_pits.shape[0]
        queries_raw = raw_pits[indices]
    else:
        queries_raw = raw_pits[:num_queries]
        
    # Save float32 queries directly for Pithos
    queries_float = queries_raw.astype(np.float32)
    
    # Partition queries into 8 families (0 to 7)
    families = np.arange(num_queries, dtype=np.int32) % 8
    np.random.seed(42)
    np.random.shuffle(families)
    
    # Set distance threshold for each family.
    # For DINOv3 + Lunar LoRA, target threshold is gestaucht to around 40 bits.
    thresholds = np.full(num_queries, 40, dtype=np.int32)
    
    # Save arrays for the verification phase
    np.save("temp/benchmark_data/queries.npy", queries_float)
    np.save("temp/benchmark_data/families.npy", families)
    np.save("temp/benchmark_data/thresholds.npy", thresholds)
    
    print(f"Generated {num_queries} queries partitioned across 8 families.")
    print("Queries saved successfully as float arrays.")

if __name__ == "__main__":
    main()
