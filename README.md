# Pithos Vector Search Engine

A high-performance, Ahead-of-Time (AOT) compiled, dimension-agnostic vector search engine written in **Java 25**, optimized for **Matryoshka-structured binary embeddings** at planetary scale, and compiled into a native shared library (`.dylib` / `.so`) via **GraalVM Native Image**.

Pithos achieves its speed by collapsing abstraction boundaries between language runtimes, the operating system, and hardware execution models. It bypasses garbage collection entirely, mapping memory-bandwidth-bound datasets off-heap using the Java Foreign Function & Memory (FFM) API (Project Panama) and POSIX-aligned virtual memory mapping (`mmap`).

---

## Architectural Principles & Core Innovations

Pithos is built on the premise that the database is a physical extension of the embedding model itself. Both share a mathematical contract established during model training:

```
                  [ Raw Input Vector x ]
                            │
                            ▼
        [ Rademacher Preconditioning: z = x * D_pre ]
                            │
                            ▼
      [ Block Walsh-Hadamard Rotation: z = H_BD * z ]
                            │
                            ▼
           [ 1-Bit Binarization: b = sign(z) ]
                            │
                            ▼
               [ Cascaded Hierarchical Scan ]
               ├── Gate 1: Tombstone & Attribute Mask
               ├── Gate 2: Quantization Entropy Gate (QEG)
               └── Gate 3: XOR-Popcount Cascade (Tiers 0..T)
```

### 1. Isomorphic Transformation & Matryoshka Tiers
Before binarization, raw input embeddings are transformed using a structured orthogonal mapping designed to preserve angular distance geometry:
- **Rademacher Preconditioning ($D_{\mathrm{pre}}$):** A stochastic sign-flipping diagonal operator that whitens coordinate covariance and prevents signal entropy leakage:
  $$D_{\mathrm{pre}} = \mathrm{diag}(d_1, \dots, d_D) \quad \text{where } d_j \in \{-1, 1\} \text{ are independent Rademacher variables.}$$
  For an input vector $x \in \mathbb{R}^D$, preconditioning is computed as the Hadamard product:
  $$x' = x \odot d$$
- **Block-Diagonal Walsh-Hadamard Rotation ($H_{\mathrm{BD}}$):** Rotation is computed as a direct sum ($\oplus$) of independent Sylvester-Hadamard matrices corresponding to each Matryoshka tier width $\Delta s_k = s_k - s_{k-1}$:
  $$H_{\mathrm{BD}} = \bigoplus_{k=1}^T H_{\Delta s_k}$$
  where each Sylvester-Hadamard matrix $H_n$ is normalized by $1 / \sqrt{n}$ to remain orthogonal, and is recursively defined as:
  $$H_{2^m} = \frac{1}{\sqrt{2}} \begin{bmatrix} H_{2^{m-1}} & H_{2^{m-1}} \\ H_{2^{m-1}} & -H_{2^{m-1}} \end{bmatrix} \quad \text{with } H_1 = [1].$$
- **Kronecker Fallback:** For arbitrary block sizes that are not powers of two, Pithos factorizes the width $\Delta s_k$ into $u \times v$ (where $u = 2^m$ is the largest power of two dividing or matching the dimension factor) and applies the Kronecker product ($\otimes$):
  $$H_{\Delta s_k} = H_u \otimes \Omega_v$$
  where $\Omega_v$ is a deterministic orthonormal Discrete Cosine Transform (DCT) matrix of size $v \times v$, defined as:
  $$\Omega_{v}(p, q) = \sqrt{\frac{2 - \delta_{p,0}}{v}} \cos\left( \frac{\pi (2q + 1) p}{2v} \right) \quad \text{for } p,q \in \{0, \dots, v-1\}$$
  where $\delta_{p,0}$ is the Kronecker delta.

### 2. SVD-Driven Spectral Truncation
At load time, Pithos accepts the model's frozen adapter weight matrix $W \in \mathbb{R}^{D \times r}$. The engine executes a native, zero-dependency **Jacobi SVD solver** to compute singular values $\sigma_1, \dots, \sigma_D$ by applying iterative Jacobi rotations to diagonalize the covariance matrix $C = W^T W$. This allows reconstruction of the cumulative spectral energy distribution $\Phi(k)$:
$$\Phi(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{j=1}^{\min(D,r)} \sigma_j^2}$$
Given a target information budget $\tau \in (0, 1]$, Pithos computes the F1-optimal pruning tier boundary:
$$\mathcal{T}(S,\tau) = \min \{ k \mid \Phi(s_k) \ge \tau \}$$
All database columns matching tiers $k > \mathcal{T}(S,\tau)$ are bypassed during search, saving memory bus I/O bandwidth.

### 3. Zero-Overhead Columnar Multi-Tier Layout
Pithos abandons flat 64-byte file layouts in favor of raw binary tier columns:
- **Positional Identity Mapping:** Records do not store explicit identifiers inside tier files. The index offset $i$ serves as the global identity across `tier_0.bin` to `tier_n.bin`.
- **Address Resolution:** For tier $k$, the byte address of record $i$'s binarized words is calculated in $O(1)$:
  $$\operatorname{Address}(i,k) = \operatorname{Base}_k + i \cdot \frac{\Delta s_k}{8}$$
  where $\operatorname{Base}_k$ is the memory segment offset for tier $k$.
- **Attribute & Tombstone Columns:** Deletions ($T_i$) and validity masks ($M_i$) are stored in a dedicated `metadata.bin` file of size $N \times 8$ bytes, updated in-place without physical layout reorganization.

### 4. Three-Gate Cascaded Read-Path
Query vectors are binarized as $b(q) = \operatorname{sign}(z(q)) \in \{0, 1\}^D$ and cascaded through registration gates to prevent unneeded memory-bus transfers:
- **Gate 1 (Liveliness):** Skips record if the tombstone bit is set ($T_i = 1$) or the attribute validity bit is missing ($M_i = 0$).
- **Gate 2 (Quantization Entropy Gate - QEG):** Evaluates macro-topography in the first tier (Tier 0). Specifically, if the most significant bit (MSB, bit 63) of the first 64-bit word of Tier 0 of the record is 0:
  $$\operatorname{MSB}(t_i^{(0)}) = 0$$
  the record is classified as flat terrain and the search early-terminates.
- **Gate 3 (XOR-Popcount Cascade):** Computes partial Hamming distance tier-by-tier up to active tier $T$:
  $$\mathcal{D}_H^{(k)}(b_i, b(q)) = \sum_{d=1}^{s_k} b_{i,d} \oplus b_{q,d}$$
  If at any tier $k \le T$, the accumulated distance $\mathcal{D}_H^{(k)}$ exceeds the query threshold $T_q$, the sweep terminates before reading subsequent tier files from memory.

### 5. Multi-Family Resonant Voting
For planetary-scale anomaly verification, Pithos implements a lock-free multi-family resonant voting schema. Given a set of queries $Q = \{q_1, \dots, q_M\}$ split into $F$ families (each query $q_j$ assigned family $f_j \in \{0, \dots, F-1\}$ and threshold $T_j$):
- Each worker thread builds a thread-local bitmask of resonant family votes $V_i$ for record $i$:
  $$V_i = \bigvee_{j=1}^M \mathbb{I}\left( \mathcal{D}_H^{(T)}(b_i, b(q_j)) \le T_j \right) \cdot 2^{f_j}$$
- The thread-local bitmasks are merged across worker pools using a bitwise OR operation:
  $$V_i^{\text{merged}} = \bigvee_{w=1}^{N_{\text{workers}}} V_{i,w}$$
- A record $i$ is returned as a resonant match if the total number of families voting for it meets the vote threshold $K_{\text{vote}}$:
  $$\operatorname{popcount}(V_i^{\text{merged}}) \ge K_{\text{vote}} \quad \text{where } K_{\text{vote}} = 5 \text{ (out of } F=8 \text{ families)}.$$

---

## Directory Structure

```
.
├── pom.xml                 # Maven configuration (dimension-agnostic pithos packaging)
├── Dockerfile              # Multi-stage compile environment with GraalVM JDK 25 and GCC
├── README.md               # This file
├── build.sh                # Docker build script (exports compiled Linux library)
├── test_client.c           # C verification client calling Pithos float C-API
├── verify_scale.sh         # Scale stress-test wrapper script
├── benchmark.py            # Python ctypes performance sweep and verification
├── run_real_verification.py# Real-data MNIST/DINOv3 verification pipeline
└── src
    ├── main
    │   └── java
    │       └── org
    │           └── pithos
    │               ├── CApi.java           # GraalVM Native C FFI Bridge Entrypoints
    │               ├── DistanceMetric.java # Unrolled popcount Hamming calculations
    │               ├── FlatIndex.java      # Multi-tier Disruptor- & Unsafe-optimized index
    │               ├── Index.java          # Core Index interface
    │               ├── TransformOperator.java# Jacobi SVD, Rademacher preconditioning & block FWHT
    │               ├── VectorDb.java       # DB manager and multi-tier index compiler
    │               └── VectorRecord.java   # Dimension-agnostic record representation
    └── test
        └── java
            └── org
                └── pithos
                    └── VectorDbTest.java   # Unit tests for SVD, FWHT, and compiled query logic
```

---

## C-API Reference

The compiled native library exposes the following dynamic C interfaces:

```c
// Creates a GraalVM isolate context for JVM execution
int graal_create_isolate(graal_isolate_params_t* params, graal_isolate_t** isolate, graal_isolatethead_t** thread);

// Initializes the Pithos database coordinator
int vdb_init(graal_isolatethead_t* thread);

// Maps an existing multi-tier database off-heap (equal spectral distribution fallback)
int vdb_load_index(graal_isolatethead_t* thread, char* name, char* path);

// Maps an existing database and supplies frozen LoRA weight matrices to compute spectral energy
int vdb_load_index_with_weights(graal_isolatethead_t* thread, char* name, char* path, float* weights, int loraDim);

// Retrieves database metadata attributes (dimension, size, planet settings, tiers count)
int vdb_get_info(graal_isolatethead_t* thread, char* indexName, int* outDimension, long long* outSize, char* outPlanetId, long long* outPlanetRadius, int* outTiersCount);

// Compiles raw float records into a multi-tier database file layout
int vdb_compile_index_file(graal_isolatethead_t* thread, char* path, byte planetId, long long planetRadius, int dimension, int* tiers, int numTiers, long long* ids, float* vectors, int numRecords);

// Batch KNN search over raw float vectors
int vdb_batch_search(graal_isolatethead_t* thread, char* indexName, float* queries, int numQueries, int k, long long* outIds, int* outDistances);

// Multi-Family Resonant Voting search over raw float queries
long long vdb_query_planetary_grid(graal_isolatethead_t* thread, char* indexName, float* queries, int* queryFamilies, int* queryThresholds, int numQueries, char* votingMask);

// Sets the parallel Disruptor chunk sweep size
int vdb_set_chunk_size(graal_isolatethead_t* thread, char* indexName, long long chunkSize);

// Sets the active energy budget budget (0.0 to 1.0) to prune lower tiers dynamically
int vdb_set_energy_budget(graal_isolatethead_t* thread, char* indexName, double tau);

// Returns record size of mapped index
long long vdb_size(graal_isolatethead_t* thread, char* indexName);

// Drops/closes an index
int vdb_drop_index(graal_isolatethead_t* thread, char* indexName);

// Shuts down database and frees mapped pages
int vdb_close(graal_isolatethead_t* thread);

// Tears down GraalVM isolate thread
int graal_tear_down_isolate(graal_isolatethead_t* thread);
```

---

## System Verification & Performance Results
 
### 1. Real-Data Verification (Real Lunar Pits Dataset, Mode B — Fine-Tuned DINOv3 + Lunar LoRA Adapter)
Pithos transforms, binarizes, and indexes raw float vectors into 384-bit Matryoshka-structured binary embeddings. Evaluation is performed natively on the macOS host to eliminate virtualization context-switch latencies.
 
- **Target Class:** Lunar Pit/Cave Entrance Anchor
- **Optimal Decision Boundary (Hamming Threshold):** $\le 49$ bits (dynamically F1-optimized)
- **Precision:** **66.56%**
- **Recall:** **84.65%**
- **F1-Score:** **74.52%**
- **Hamming Distance Distribution:**
  - Query vs Target Class (Pits): Mean **131.22 bits** ($\sigma$: **76.84 bits**)
  - Query vs Background Class (Mondgelände): Mean **197.32 bits** ($\sigma$: **75.01 bits**)
 
#### Confusion Matrix
* **True Positives (TP):** 140,118
* **False Positives (FP):** 70,390
* **False Negatives (FN):** 25,410
* **True Negatives (TN):** 764,082
 
#### Search Execution Performance (Host-Native macOS)
- **Scan Latency:** **44.67 ms** for 1,000,000 records (278 queries)
- **Throughput:** **6,223.72 million vectors / sec (MVPS)** (using lock-free multi-family resonant voting)
 
### 2. High-Performance Native Performance vs. Baselines & Virtualization
Bypassing Docker Desktop's virtualization layer and running natively on the macOS host avoids hypervisor scheduling overheads and OS context switching on BlockingWaitStrategy locks. Furthermore, Pithos's binarized projection and 3-Gate early-exit cascade yield enormous speedups over exact float scan baselines on the same 1,000,000 replicated lunar vector database:
- **Sequential JIT Compiled Baseline:** **4.52 MVPS** (simulated float L2 scan)
- **FAISS Flat L2 Baseline (CPU Native):** **75.38 MVPS**
- **Docker VM Pithos Throughput:** **955.34 MVPS**
- **Host-Native macOS Pithos Throughput:** **6,223.72 MVPS** (a **~6.5x speedup** over Docker, and a **~82.6x speedup** over native FAISS Flat L2)
 
### 3. Visual Charts (Vector Anomaly Distribution & Throughput Analysis)
Below are the dynamically generated SVG charts visualizing our classification metrics and execution performance:
 
#### Hamming Distance Distribution:
![Hamming Distance Distribution](assets/distribution_plot.svg)
 
#### Throughput Comparison:
![Throughput Comparison](assets/throughput_comparison.svg)

## Build & Run

### 1. Compile & Build (Native macOS)
Ensure you have the workspace-bundled **GraalVM JDK 25** and **Maven** installed, then set `JAVA_HOME` and compile:
```bash
export JAVA_HOME=/Users/finnhertsch/projects/lcvk/graalvm/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
mvn clean package
```
This executes all unit tests (isomorphic matrices, Kronecker fallbacks, SVD solvers, multi-tier indexing) and compiles `libpithos.dylib` inside `target/` and the root directory.

### 2. Running Scale Benchmark
```bash
export JAVA_HOME=/Users/finnhertsch/projects/lcvk/graalvm/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
.venv/bin/python benchmark.py
```
This dynamically compiles a scale dataset of $500,000$ float records, maps them off-heap, loads target weights matrix to compute SVD energy densities, and runs query sweeps.

### 3. Real-Data Verification
```bash
export JAVA_HOME=/Users/finnhertsch/projects/lcvk/graalvm/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
.venv/bin/python run_real_verification.py
```
This runs the full Lunar Pit / DINOv3 + Lunar LoRA pipeline, ingests raw float vectors, optimizes Hamming classification thresholds, and validates Pithos precision and F1-score.

### 4. Running Baseline Benchmarks
```bash
.venv/bin/python benchmark_baselines.py
```
This runs both the FAISS Flat Index L2 baseline and a simulated single-threaded JIT sequential scan over the actual 1,000,000 replicated lunar vector database, saving baseline results to `baselines_metrics.json`.
