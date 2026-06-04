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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadFactory;
import java.util.stream.IntStream;

import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.WorkHandler;
import com.lmax.disruptor.YieldingWaitStrategy;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;

import sun.misc.Unsafe;
import java.lang.reflect.Field;

/**
 * Hardware-optimized Flat Index utilizing sun.misc.Unsafe for direct memory access
 * and LMAX Disruptor for lock-free execution mapping.
 *
 * File Structure (64-byte Cache-Line Aligned):
 * - 64-byte Header (Bytes 0-63)
 * - N * 64-byte aligned records:
 *   - Offset 0 (8 bytes): id (long)
 *   - Offset 8 (48 bytes): vector data (6 longs / 384 bits)
 *   - Offset 56 (8 bytes): metadata (long)
 */
public class FlatIndex implements Index {

    private static final Unsafe UNSAFE;
    static {
        try {
            Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
            theUnsafe.setAccessible(true);
            UNSAFE = (Unsafe) theUnsafe.get(null);
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize sun.misc.Unsafe", e);
        }
    }

    private final MemorySegment segment;
    private final long size;

    // Reusable, thread-safe Disruptor pool
    private final Disruptor<RangeEvent> disruptor;
    private final RingBuffer<RangeEvent> ringBuffer;
    private final int numWorkers;
    private volatile long chunkSize = 20000;

    public void setChunkSize(long chunkSize) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("Chunk size must be greater than zero");
        }
        this.chunkSize = chunkSize;
    }

    /**
     * Pojo capturing a segment block scanning event for the Disruptor.
     */
    public static class RangeEvent {
        public long startIdx;
        public long endIdx;
        public long[][] queries;
        public int k;
        public int[][][] threadLocalDists;
        public long[][][] threadLocalIds;
        public boolean isVoting;
        public CountDownLatch latch;
        
        // Voting parameters
        public int[] families;
        public int[] thresholds;
        public MemorySegment[] threadLocalMasks;

        public void setKnn(long startIdx, long endIdx, long[][] queries, int k, 
                           long[][][] threadLocalIds, int[][][] threadLocalDists, CountDownLatch latch) {
            this.startIdx = startIdx;
            this.endIdx = endIdx;
            this.queries = queries;
            this.k = k;
            this.threadLocalIds = threadLocalIds;
            this.threadLocalDists = threadLocalDists;
            this.isVoting = false;
            this.latch = latch;
        }

        public void setVoting(long startIdx, long endIdx, long[][] queries, int[] families, int[] thresholds, 
                              MemorySegment[] threadLocalMasks, CountDownLatch latch) {
            this.startIdx = startIdx;
            this.endIdx = endIdx;
            this.queries = queries;
            this.families = families;
            this.thresholds = thresholds;
            this.threadLocalMasks = threadLocalMasks;
            this.isVoting = true;
            this.latch = latch;
        }
    }

    /**
     * WorkHandler assigning parallel range blocks dynamically to execution threads.
     */
    private static class RangeWorkHandler implements WorkHandler<RangeEvent> {
        private final FlatIndex index;
        private final int threadId;

        public RangeWorkHandler(FlatIndex index, int threadId) {
            this.index = index;
            this.threadId = threadId;
        }

        @Override
        public void onEvent(RangeEvent event) throws Exception {
            try {
                if (event.isVoting) {
                    index.executeVotingRange(event.startIdx, event.endIdx, event.queries, event.families, event.thresholds, event.threadLocalMasks[threadId]);
                } else {
                    index.executeKnnRange(event.startIdx, event.endIdx, event.queries, event.k, 
                                          event.threadLocalIds[threadId], event.threadLocalDists[threadId]);
                }
            } finally {
                event.latch.countDown();
            }
        }
    }

    public FlatIndex(MemorySegment segment) {
        if (segment == null) {
            throw new IllegalArgumentException("MemorySegment cannot be null");
        }
        this.segment = segment;
        if (segment.byteSize() <= 64) {
            this.size = 0;
        } else {
            // Header is 64 bytes, each record is 64 bytes
            this.size = (segment.byteSize() - 64) / 64;
        }

        this.numWorkers = Runtime.getRuntime().availableProcessors();
        
        ThreadFactory threadFactory = r -> {
            Thread t = new Thread(r, "lcvk-disruptor-worker");
            t.setDaemon(true);
            return t;
        };

        this.disruptor = new Disruptor<>(
                RangeEvent::new,
                1024,
                threadFactory,
                ProducerType.SINGLE,
                new YieldingWaitStrategy()
        );

        RangeWorkHandler[] handlers = new RangeWorkHandler[numWorkers];
        for (int i = 0; i < numWorkers; i++) {
            handlers[i] = new RangeWorkHandler(this, i);
        }
        this.disruptor.handleEventsWithWorkerPool(handlers);
        this.ringBuffer = this.disruptor.start();
    }

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
        throw new UnsupportedOperationException("Insert is not supported on read-only memory-mapped Index.");
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

        long numRecords = this.size;
        if (numRecords == 0) {
            List<SearchResult>[] empty = new List[queries.length];
            Arrays.fill(empty, List.of());
            return empty;
        }

        int numQueries = queries.length;

        // Allocate thread-local structures to hold top-K results
        long[][][] threadLocalIds = new long[numWorkers][numQueries][k];
        int[][][] threadLocalDists = new int[numWorkers][numQueries][k];

        // Initialize distances to infinity
        for (int w = 0; w < numWorkers; w++) {
            for (int q = 0; q < numQueries; q++) {
                Arrays.fill(threadLocalDists[w][q], Integer.MAX_VALUE);
            }
        }

        // Divide work into lock-free chunk events
        long currentChunkSize = this.chunkSize;
        long numChunks = (numRecords + currentChunkSize - 1) / currentChunkSize;
        CountDownLatch latch = new CountDownLatch((int) numChunks);

        for (long c = 0; c < numChunks; c++) {
            long startIdx = c * currentChunkSize;
            long endIdx = Math.min(startIdx + currentChunkSize, numRecords);

            long sequence = ringBuffer.next();
            try {
                RangeEvent event = ringBuffer.get(sequence);
                event.setKnn(startIdx, endIdx, queries, k, threadLocalIds, threadLocalDists, latch);
            } finally {
                ringBuffer.publish(sequence);
            }
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Search execution was interrupted", e);
        }

        // Merge thread-local results for each query
        List<SearchResult>[] finalResults = new List[numQueries];
        for (int q = 0; q < numQueries; q++) {
            List<SearchResult> merged = new ArrayList<>();
            for (int w = 0; w < numWorkers; w++) {
                long[] ids = threadLocalIds[w][q];
                int[] dists = threadLocalDists[w][q];
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
                return Long.compare(r1.id(), r2.id());
            });

            // Keep top K
            if (merged.size() > k) {
                finalResults[q] = new ArrayList<>(merged.subList(0, k));
            } else {
                finalResults[q] = merged;
            }
        }

        return finalResults;
    }

    private void executeKnnRange(long startIdx, long endIdx, long[][] queries, int k, long[][] myIds, int[][] myDists) {
        long baseAddr = segment.address();
        int numQueries = queries.length;
        long[] tileVector = new long[6];

        for (long i = startIdx; i < endIdx; i++) {
            // Offset calculation: (i + 1) * 64 bytes
            long baseOffset = (i + 1) << 6;
            
            // Read 64-bit ID
            long recordId = UNSAFE.getLong(baseAddr + baseOffset);

            // Read 6 longs (384-bit vector) starting from offset 8
            tileVector[0] = UNSAFE.getLong(baseAddr + baseOffset + 8);
            tileVector[1] = UNSAFE.getLong(baseAddr + baseOffset + 16);
            tileVector[2] = UNSAFE.getLong(baseAddr + baseOffset + 24);
            tileVector[3] = UNSAFE.getLong(baseAddr + baseOffset + 32);
            tileVector[4] = UNSAFE.getLong(baseAddr + baseOffset + 40);
            tileVector[5] = UNSAFE.getLong(baseAddr + baseOffset + 48);

            // Single-Pass Scan: Stream queries against loaded vector
            for (int q = 0; q < numQueries; q++) {
                int dist = DistanceMetric.HAMMING.calculate(queries[q], tileVector);

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
    }

    @Override
    public long queryPlanetaryGrid(long[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        if (queries == null || queries.length == 0) return 0;
        int numQueries = queries.length;
        long numRecords = this.size;
        if (numRecords == 0) return 0;

        long currentChunkSize = this.chunkSize;
        long numChunks = (numRecords + currentChunkSize - 1) / currentChunkSize;
        CountDownLatch latch = new CountDownLatch((int) numChunks);

        // Allocate thread-local voting masks using Arena
        Arena arena = Arena.global();
        MemorySegment[] threadLocalMasks = new MemorySegment[numWorkers];
        for (int w = 0; w < numWorkers; w++) {
            threadLocalMasks[w] = arena.allocate(numRecords);
        }

        for (long c = 0; c < numChunks; c++) {
            long startIdx = c * currentChunkSize;
            long endIdx = Math.min(startIdx + currentChunkSize, numRecords);

            long sequence = ringBuffer.next();
            try {
                RangeEvent event = ringBuffer.get(sequence);
                event.setVoting(startIdx, endIdx, queries, families, thresholds, threadLocalMasks, latch);
            } finally {
                ringBuffer.publish(sequence);
            }
        }

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Voting execution was interrupted", e);
        }

        // Merge thread-local masks and count resonant tiles in parallel using Unsafe reads
        long maskAddr = votingMask.address();
        int numThreads = Runtime.getRuntime().availableProcessors();
        long recordsPerThread = numRecords / numThreads;
        if (recordsPerThread == 0) {
            numThreads = 1;
            recordsPerThread = numRecords;
        }
        final int activeThreads = numThreads;
        final long finalRecordsPerThread = recordsPerThread;

        long[] localMaskAddrs = new long[numWorkers];
        for (int w = 0; w < numWorkers; w++) {
            localMaskAddrs[w] = threadLocalMasks[w].address();
        }

        return IntStream.range(0, activeThreads).parallel().mapToLong(t -> {
            long startIdx = t * finalRecordsPerThread;
            long endIdx = (t == activeThreads - 1) ? numRecords : (t + 1) * finalRecordsPerThread;
            long resonantCount = 0;
            for (long i = startIdx; i < endIdx; i++) {
                byte mergedVal = 0;
                for (int w = 0; w < numWorkers; w++) {
                    mergedVal |= UNSAFE.getByte(localMaskAddrs[w] + i);
                }
                UNSAFE.putByte(maskAddr + i, mergedVal);
                if (Integer.bitCount(mergedVal & 0xFF) >= 7) {
                    resonantCount++;
                }
            }
            return resonantCount;
        }).sum();
    }

    private void executeVotingRange(long startIdx, long endIdx, long[][] queries, int[] families, int[] thresholds, MemorySegment localMask) {
        long baseAddr = segment.address();
        long localMaskAddr = localMask.address();
        int numQueries = queries.length;
        long[] tileVector = new long[6];

        for (long i = startIdx; i < endIdx; i++) {
            long baseOffset = (i + 1) << 6;

            tileVector[0] = UNSAFE.getLong(baseAddr + baseOffset + 8);
            tileVector[1] = UNSAFE.getLong(baseAddr + baseOffset + 16);
            tileVector[2] = UNSAFE.getLong(baseAddr + baseOffset + 24);
            tileVector[3] = UNSAFE.getLong(baseAddr + baseOffset + 32);
            tileVector[4] = UNSAFE.getLong(baseAddr + baseOffset + 40);
            tileVector[5] = UNSAFE.getLong(baseAddr + baseOffset + 48);

            byte maskVal = 0;
            for (int q = 0; q < numQueries; q++) {
                int dist = DistanceMetric.HAMMING.calculate(queries[q], tileVector);
                if (dist <= thresholds[q]) {
                    maskVal |= (byte) (1 << families[q]);
                }
            }
            UNSAFE.putByte(localMaskAddr + i, maskVal);
        }
    }

    @Override
    public int getDimension() {
        return 384;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public void close() {
        if (disruptor != null) {
            try {
                disruptor.shutdown();
            } catch (Throwable t) {
                // ignore
            }
        }
    }
}
