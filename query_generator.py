import numpy as np
from ingest_pipeline import precondition_and_quantize, pack_bits, DIMENSION

def main():
    # Load raw embeddings of sevens saved by ingest_pipeline.py
    if not os.path.exists("raw_sevens.npy"):
        print("[Error] raw_sevens.npy not found. Please run ingest_pipeline.py first.")
        sys.exit(1)
        
    raw_sevens = np.load("raw_sevens.npy")
    print(f"Loaded {raw_sevens.shape[0]} raw embeddings of digit '7'.")
    
    # Select 278 queries
    num_queries = 278
    if raw_sevens.shape[0] < num_queries:
        print(f"[Warning] Only found {raw_sevens.shape[0]} sevens. Repeating to reach {num_queries}.")
        indices = np.arange(num_queries) % raw_sevens.shape[0]
        queries_raw = raw_sevens[indices]
    else:
        queries_raw = raw_sevens[:num_queries]
        
    # Precondition and binarize queries using the exact same PolarQuant-Hadamard logic
    bits = precondition_and_quantize(queries_raw)
    packed_queries = pack_bits(bits)  # shape (278, 48)
    
    # Convert packed queries to 6 longs per query (dtype=np.int64)
    # 48 bytes is exactly 6 int64s. We can use .view(np.int64)
    queries_long = packed_queries.view(np.int64) # shape (278, 6)
    
    # Partition queries into 8 families (0 to 7)
    families = np.arange(num_queries, dtype=np.int32) % 8
    
    # Set distance threshold for each family.
    # A Hamming distance threshold of 110 (out of 384 bits) is a good balance:
    # it allows for style variations in '7's while rejecting other digits (dist ~ 192).
    thresholds = np.full(num_queries, 110, dtype=np.int32)
    
    # Save arrays for the verification phase
    np.save("queries.npy", queries_long)
    np.save("families.npy", families)
    np.save("thresholds.npy", thresholds)
    
    print(f"Generated {num_queries} queries partitioned across 8 families.")
    print("Queries packed and saved successfully.")

if __name__ == "__main__":
    import os
    import sys
    main()
