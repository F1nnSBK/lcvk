package org.lcvk.vectordb;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * High-performance Flat Index using Project Panama (FFM) MemorySegments.
 * Maps a file directly off-heap to support multi-gigabyte vector files.
 * Streams records linearly and applies SIMD Hamming distance in a parallel Single-Pass Multi-Query Scan.
 *
 * File Structure:
 * - 64-byte Header (Bytes 0-63)
 * - N * 48-byte pure flat vectors (6 longs / 384 bits each)
 * - Geographic ID is implicitly determined by the record index (0, 1, 2, ...)
 */
public class FlatIndex implements Index {
    private final MemorySegment segment;
    private final long size;

    /**
     * Instantiates FlatIndex using an existing MemorySegment.
     */
    public FlatIndex(MemorySegment segment) {
        if (segment == null) {
            throw new IllegalArgumentException("MemorySegment cannot be null");
        }
        this.segment = segment;
        if (segment.byteSize() <= 64) {
            this.size = 0;
        } else {
            // Header is 64 bytes, each vector is 48 bytes (6 longs)
            this.size = (segment.byteSize() - 64) / 48;
        }
    }

    /**
     * Factory method to map a vector file directly to virtual memory.
     * Enforces magic bytes PLAN validation.
     *
     * @param path path to the vector database file
     * @return a mapped FlatIndex instance
     * @throws IOException if mapping fails or magic bytes mismatch
     */
    public static FlatIndex mapFile(String path) throws IOException {
        Path filePath = Path.of(path);
        try (FileChannel channel = FileChannel.open(filePath, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            MemorySegment mappedSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, Arena.global());

            // Validate header magic bytes "PLAN"
            if (fileSize >= 4) {
                byte m0 = mappedSegment.get(ValueLayout.JAVA_BYTE, 0);
                byte m1 = mappedSegment.get(ValueLayout.JAVA_BYTE, 1);
                byte m2 = mappedSegment.get(ValueLayout.JAVA_BYTE, 2);
                byte m3 = mappedSegment.get(ValueLayout.JAVA_BYTE, 3);
                if (m0 != 'P' || m1 != 'L' || m2 != 'A' || m3 != 'N') {
                    throw new IllegalArgumentException("Invalid LCVK file magic: must be PLAN");
                }
            }
            return new FlatIndex(mappedSegment);
        }
    }

    @Override
    public void insert(VectorRecord record) {
        throw new UnsupportedOperationException("Insert is not supported on a read-only memory-mapped Index. Compile files offline.");
    }

    @Override
    public List<SearchResult> search(long[] query, int k) {
        List<SearchResult>[] results = batchSearch(new long[][]{query}, k);
        return results[0];
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<SearchResult>[] batchSearch(long[][] queries, int k) {
        if (queries == null || queries.length == 0) {
            return new List[0];
        }
        if (k <= 0) {
            List<SearchResult>[] empty = new List[queries.length];
            Arrays.fill(empty, List.of());
            return empty;
        }

        int numQueries = queries.length;
        long numRecords = this.size;

        // Determine thread partitioning
        int numThreads = Runtime.getRuntime().availableProcessors();
        long recordsPerThread = numRecords / numThreads;
        if (recordsPerThread == 0) {
            numThreads = 1;
            recordsPerThread = numRecords;
        }

        final int activeThreads = numThreads;
        final long finalRecordsPerThread = recordsPerThread;

        // Allocate thread-local structures to hold top-K results
        long[][][] threadLocalIds = new long[activeThreads][numQueries][k];
        int[][][] threadLocalDists = new int[activeThreads][numQueries][k];

        // Initialize distances to max value (infinity)
        for (int t = 0; t < activeThreads; t++) {
            for (int q = 0; q < numQueries; q++) {
                Arrays.fill(threadLocalDists[t][q], Integer.MAX_VALUE);
            }
        }

        // Parallel processing across thread chunks
        IntStream.range(0, activeThreads).parallel().forEach(t -> {
            long startIdx = t * finalRecordsPerThread;
            long endIdx = (t == activeThreads - 1) ? numRecords : (t + 1) * finalRecordsPerThread;

            long[][] myIds = threadLocalIds[t];
            int[][] myDists = threadLocalDists[t];

            // Allocate a reusable local buffer per thread to avoid garbage collection overhead
            long[] tileVector = new long[6];

            for (long i = startIdx; i < endIdx; i++) {
                long vectorOffset = 64 + i * 48L; // 64-byte header + i * 48 bytes
                long recordId = i; // Geographic ID is implicitly the tile index

                // Load 6 longs directly from mapped MemorySegment into L1-cached local array
                tileVector[0] = segment.get(ValueLayout.JAVA_LONG, vectorOffset);
                tileVector[1] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 8);
                tileVector[2] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 16);
                tileVector[3] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 24);
                tileVector[4] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 32);
                tileVector[5] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 40);

                // Single-Pass Scan: Stream queries against the loaded tile vector
                for (int q = 0; q < numQueries; q++) {
                    int dist = DistanceMetric.HAMMING.calculate(queries[q], tileVector);

                    // Inline top-K accumulation (avoiding PriorityQueue object allocation)
                    int[] dists = myDists[q];
                    if (dist < dists[k - 1]) {
                        long[] ids = myIds[q];
                        int pos = k - 1;
                        while (pos > 0 && dist < dists[pos - 1]) {
                            dists[pos] = dists[pos - 1];
                            ids[pos] = ids[pos - 1];
                            pos--;
                        }
                        dists[pos] = dist;
                        ids[pos] = recordId;
                    }
                }
            }
        });

        // Merge thread-local results for each query
        List<SearchResult>[] finalResults = new List[numQueries];
        for (int q = 0; q < numQueries; q++) {
            List<SearchResult> merged = new ArrayList<>();
            for (int t = 0; t < activeThreads; t++) {
                long[] ids = threadLocalIds[t][q];
                int[] dists = threadLocalDists[t][q];
                for (int i = 0; i < k; i++) {
                    if (dists[i] != Integer.MAX_VALUE) {
                        merged.add(new SearchResult(ids[i], dists[i]));
                    }
                }
            }

            // Sort merged matches by Hamming distance (ascending)
            merged.sort((r1, r2) -> {
                int cmp = Integer.compare(r1.score(), r2.score());
                if (cmp != 0) return cmp;
                return Long.compare(r1.id(), r2.id()); // deterministic tie-breaker
            });

            // Keep top K
            if (merged.size() > k) {
                finalResults[q] = merged.subList(0, k);
            } else {
                finalResults[q] = merged;
            }
        }

        return finalResults;
    }

    @Override
    public long queryPlanetaryGrid(long[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        if (queries == null || queries.length == 0) return 0;
        int numQueries = queries.length;
        long numRecords = this.size;

        int numThreads = Runtime.getRuntime().availableProcessors();
        long recordsPerThread = numRecords / numThreads;
        if (recordsPerThread == 0) {
            numThreads = 1;
            recordsPerThread = numRecords;
        }

        final int activeThreads = numThreads;
        final long finalRecordsPerThread = recordsPerThread;

        // Perform parallel scan over disjoint sections of the voting mask
        IntStream.range(0, activeThreads).parallel().forEach(t -> {
            long startIdx = t * finalRecordsPerThread;
            long endIdx = (t == activeThreads - 1) ? numRecords : (t + 1) * finalRecordsPerThread;

            long[] tileVector = new long[6];

            for (long i = startIdx; i < endIdx; i++) {
                long vectorOffset = 64 + i * 48L;

                // Load vector (zero allocations)
                tileVector[0] = segment.get(ValueLayout.JAVA_LONG, vectorOffset);
                tileVector[1] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 8);
                tileVector[2] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 16);
                tileVector[3] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 24);
                tileVector[4] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 32);
                tileVector[5] = segment.get(ValueLayout.JAVA_LONG, vectorOffset + 40);

                byte maskVal = 0;
                for (int q = 0; q < numQueries; q++) {
                    int dist = DistanceMetric.HAMMING.calculate(queries[q], tileVector);
                    if (dist <= thresholds[q]) {
                        maskVal |= (byte) (1 << families[q]);
                    }
                }
                votingMask.set(ValueLayout.JAVA_BYTE, i, maskVal);
            }
        });

        // Parallel reduction to count resonant tiles (>= 7 active bits set)
        return IntStream.range(0, activeThreads).parallel().mapToLong(t -> {
            long startIdx = t * finalRecordsPerThread;
            long endIdx = (t == activeThreads - 1) ? numRecords : (t + 1) * finalRecordsPerThread;
            long resonantCount = 0;
            for (long i = startIdx; i < endIdx; i++) {
                byte val = votingMask.get(ValueLayout.JAVA_BYTE, i);
                int bitsSet = Integer.bitCount(val & 0xFF);
                if (bitsSet >= 7) {
                    resonantCount++;
                }
            }
            return resonantCount;
        }).sum();
    }

    @Override
    public int getDimension() {
        return 384;
    }

    @Override
    public long size() {
        return size;
    }
}
