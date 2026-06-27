#!/usr/bin/env python3
import numpy as np
import time
import sys
import os

# Add parent directory to path for benchmark import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import PithosMIDB


def benchmark_cuda_vs_cpu():
    """Benchmark CUDA vs CPU performance for various batch sizes."""
    
    print("\n" + "=" * 70)
    print("PITHOS CUDA PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    # Initialize engine
    engine = PithosMIDB()
    
    # Create test index
    num_records = 10000
    dimension = 384
    
    print(f"\nGenerating test index with {num_records:,} records (D={dimension})...")
    ids = np.arange(num_records, dtype=np.int64)
    vectors = np.random.randn(num_records, dimension).astype(np.float32)
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    
    index_path = "temp/pithos_cuda_test"
    status = engine.compile_index_file(index_path, 1, 1737400, dimension, 
                                       np.array([64, 128, 256, 384], dtype=np.int32), 
                                       ids, vectors)
    if status != 0:
        print(f"Failed to compile index: {status}")
        return
    
    # Load index
    mock_weights = np.random.randn(dimension, dimension).astype(np.float32)
    status = engine.load_index("cuda_test", index_path, mock_weights, dimension)
    if status != 0:
        print(f"Failed to load index: {status}")
        return
    
    print("Index loaded successfully.")
    
    # Check CUDA availability
    cuda_available = engine.cuda_is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA not available. Running CPU-only benchmark.")
        run_cpu_benchmark(engine, dimension)
        cleanup(engine, index_path)
        return
    
    # Initialize CUDA
    engine.cuda_init(0)
    print("CUDA initialized successfully.")
    
    # Test different batch sizes
    batch_sizes = [10, 50, 100, 200, 500, 1000]
    
    print(f"\n{'Batch Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Status':<10}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        queries = np.random.randn(batch_size, dimension).astype(np.float32)
        
        # CPU Benchmark
        cpu_time = measure_cpu_search(engine, queries, 10)
        
        # GPU Benchmark
        gpu_time = measure_gpu_search(engine, queries, 10)
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        status = "OK" if speedup > 1.0 else "SLOW"
        
        print(f"{batch_size:<12} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x {status:<10}")
    
    print("-" * 60)
    
    # Voting benchmark
    print("\nRunning planetary grid voting benchmark...")
    run_voting_benchmark(engine, dimension, num_records)
    
    # Cleanup
    cleanup(engine, index_path)


def measure_cpu_search(engine, queries, k):
    """Measure CPU batch search time."""
    num_tries = 3
    times = []
    
    for _ in range(num_tries):
        start = time.perf_counter()
        engine.batch_search("cuda_test", queries, k)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return np.median(times)


def measure_gpu_search(engine, queries, k):
    """Measure GPU batch search time."""
    num_tries = 3
    times = []
    
    for _ in range(num_tries):
        start = time.perf_counter()
        engine.cuda_batch_search("cuda_test", queries, k)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return np.median(times)


def run_cpu_benchmark(engine, dimension):
    """Run CPU-only benchmark for reference."""
    batch_sizes = [10, 50, 100, 200, 500, 1000]
    
    print(f"\n{'Batch Size':<12} {'Time (ms)':<12}")
    print("-" * 30)
    
    for batch_size in batch_sizes:
        queries = np.random.randn(batch_size, dimension).astype(np.float32)
        cpu_time = measure_cpu_search(engine, queries, 10)
        print(f"{batch_size:<12} {cpu_time:<12.2f}")
    
    print("-" * 30)


def run_voting_benchmark(engine, dimension, num_records):
    """Benchmark multi-family resonant voting."""
    num_queries = 100
    num_families = 8
    
    # Generate queries
    queries = np.random.randn(num_queries, dimension).astype(np.float32)
    
    # Generate families and thresholds
    families = np.arange(num_families, dtype=np.int32)
    thresholds = np.array([40] * num_queries, dtype=np.int32)
    
    # CPU voting
    cpu_start = time.perf_counter()
    voting_mask_cpu = np.zeros(num_records, dtype=np.uint8)
    engine.query_planetary_grid("cuda_test", queries, families, thresholds, voting_mask_cpu)
    cpu_time = (time.perf_counter() - cpu_start) * 1000
    
    # GPU voting
    gpu_start = time.perf_counter()
    voting_mask_gpu = np.zeros(num_records, dtype=np.uint8)
    engine.cuda_query_planetary_grid("cuda_test", queries, families, thresholds, voting_mask_gpu)
    gpu_time = (time.perf_counter() - gpu_start) * 1000
    
    # Verify results match
    matches = np.array_equal(voting_mask_cpu, voting_mask_gpu)
    
    print(f"\nVoting Benchmark (Q={num_queries}, F={num_families}):")
    print(f"  CPU: {cpu_time:.2f} ms")
    print(f"  GPU: {gpu_time:.2f} ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x" if gpu_time > 0 else "  Speedup: N/A")
    print(f"  Results match: {matches}")


def cleanup(engine, index_path):
    """Clean up test files."""
    engine.close()
    
    # Remove index files
    for ext in ["", "_ids.bin", "_metadata.bin"]:
        path = index_path + ext
        if os.path.exists(path):
            os.remove(path)
    
    k = 0
    while True:
        path = f"{index_path}_tier_{k}.bin"
        if os.path.exists(path):
            os.remove(path)
            k += 1
        else:
            break


if __name__ == "__main__":
    benchmark_cuda_vs_cpu()
