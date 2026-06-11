package org.pithos;

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

import com.lmax.disruptor.BlockingWaitStrategy;
import com.lmax.disruptor.ExceptionHandler;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.WorkHandler;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;

import sun.misc.Unsafe;
import java.lang.reflect.Field;

/**
 * Dimension-agnostic high-performance multi-tier index.
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

    private final MemorySegment baseSegment;
    private final MemorySegment idsSegment;
    private final MemorySegment[] tierSegments;
    private final MemorySegment metadataSegment;

    private final byte planetId;
    private final long planetRadius;
    private final int dimension;
    private final int numTiers;
    private final int[] tiers;
    private final long size;

    private final TransformOperator transformOperator;
    private final float[] cumulativeEnergy;
    private double targetEnergyBudget = 0.90; // Default energy threshold tau

    // Disruptor context
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

    public void setTargetEnergyBudget(double tau) {
        if (tau <= 0.0 || tau > 1.0) {
            throw new IllegalArgumentException("Energy budget tau must be in (0, 1]");
        }
        this.targetEnergyBudget = tau;
    }

    public static class RangeEvent {
        public long startIdx;
        public long endIdx;
        public float[][] queries;
        public int k;
        public int[][][] threadLocalDists;
        public long[][][] threadLocalIds;
        public boolean isVoting;
        public CountDownLatch latch;

        // Voting parameters
        public int[] families;
        public int[] thresholds;
        public MemorySegment[] threadLocalMasks;

        public void setKnn(long startIdx, long endIdx, float[][] queries, int k,
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

        public void setVoting(long startIdx, long endIdx, float[][] queries, int[] families, int[] thresholds,
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

    public FlatIndex(MemorySegment baseSegment, MemorySegment idsSegment, MemorySegment[] tierSegments, MemorySegment metadataSegment,
                     byte planetId, long planetRadius, int dimension, int numTiers, int[] tiers, long size, float[] cumulativeEnergy) {
        this.baseSegment = baseSegment;
        this.idsSegment = idsSegment;
        this.tierSegments = tierSegments;
        this.metadataSegment = metadataSegment;
        this.planetId = planetId;
        this.planetRadius = planetRadius;
        this.dimension = dimension;
        this.numTiers = numTiers;
        this.tiers = tiers;
        this.size = size;
        this.cumulativeEnergy = cumulativeEnergy;

        this.transformOperator = new TransformOperator(dimension, tiers);

        this.numWorkers = Runtime.getRuntime().availableProcessors();
        ThreadFactory threadFactory = r -> {
            Thread t = new Thread(r, "pithos-disruptor-worker");
            t.setDaemon(true);
            return t;
        };

        this.disruptor = new Disruptor<>(
                RangeEvent::new,
                1024,
                threadFactory,
                ProducerType.SINGLE,
                new BlockingWaitStrategy()
        );
        this.disruptor.setDefaultExceptionHandler(new ExceptionHandler<RangeEvent>() {
            @Override
            public void handleEventException(Throwable ex, long sequence, RangeEvent event) {
                if (ex instanceof InterruptedException || 
                    ex.getCause() instanceof InterruptedException ||
                    (ex.getMessage() != null && ex.getMessage().contains("InterruptedException"))) {
                    return;
                }
                ex.printStackTrace();
            }

            @Override
            public void handleOnStartException(Throwable ex) {
                ex.printStackTrace();
            }

            @Override
            public void handleOnShutdownException(Throwable ex) {
                if (!(ex instanceof InterruptedException) && !(ex.getCause() instanceof InterruptedException)) {
                    ex.printStackTrace();
                }
            }
        });

        RangeWorkHandler[] handlers = new RangeWorkHandler[numWorkers];
        for (int i = 0; i < numWorkers; i++) {
            handlers[i] = new RangeWorkHandler(this, i);
        }
        this.disruptor.handleEventsWithWorkerPool(handlers);
        this.ringBuffer = this.disruptor.start();
    }

    public static FlatIndex mapFile(String basePath, float[] weights, int loraDim) throws IOException {
        Path mainPath = Path.of(basePath);
        if (!mainPath.toFile().exists()) {
            throw new IOException("Base file path does not exist: " + basePath);
        }

        // 1. Read base file and 64-byte Header
        MemorySegment mappedBase;
        try (FileChannel channel = FileChannel.open(mainPath, StandardOpenOption.READ)) {
            mappedBase = channel.map(FileChannel.MapMode.READ_ONLY, 0, 64, Arena.global());
        }

        // Validate Header Magic
        byte m0 = mappedBase.get(ValueLayout.JAVA_BYTE, 0);
        byte m1 = mappedBase.get(ValueLayout.JAVA_BYTE, 1);
        byte m2 = mappedBase.get(ValueLayout.JAVA_BYTE, 2);
        byte m3 = mappedBase.get(ValueLayout.JAVA_BYTE, 3);
        if (m0 != 'P' || m1 != 'L' || m2 != 'A' || m3 != 'N') {
            throw new IllegalArgumentException("Invalid file magic: must be PLAN");
        }

        byte planetId = mappedBase.get(ValueLayout.JAVA_BYTE, 4);
        long totalRecords = mappedBase.get(ValueLayout.JAVA_LONG_UNALIGNED, 5);
        long planetRadius = mappedBase.get(ValueLayout.JAVA_LONG_UNALIGNED, 13);
        int dimension = mappedBase.get(ValueLayout.JAVA_INT_UNALIGNED, 21);
        int numTiers = mappedBase.get(ValueLayout.JAVA_INT_UNALIGNED, 25);

        int[] tiers = new int[numTiers];
        for (int i = 0; i < numTiers; i++) {
            tiers[i] = mappedBase.get(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4));
        }

        // 2. Map ID file
        Path idsPath = Path.of(basePath + "_ids.bin");
        MemorySegment idsSegment;
        try (FileChannel channel = FileChannel.open(idsPath, StandardOpenOption.READ)) {
            idsSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * 8, Arena.global());
        }

        // 3. Map metadata file
        Path metadataPath = Path.of(basePath + "_metadata.bin");
        MemorySegment metadataSegment;
        try (FileChannel channel = FileChannel.open(metadataPath, StandardOpenOption.READ)) {
            metadataSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * 8, Arena.global());
        }

        // 4. Map Tier files
        MemorySegment[] tierSegments = new MemorySegment[numTiers];
        int prevBound = 0;
        for (int k = 0; k < numTiers; k++) {
            int width = tiers[k] - prevBound;
            Path tierPath = Path.of(basePath + "_tier_" + k + ".bin");
            try (FileChannel channel = FileChannel.open(tierPath, StandardOpenOption.READ)) {
                tierSegments[k] = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * (width / 8), Arena.global());
            }
            prevBound = tiers[k];
        }

        // 5. Compute or estimate cumulative energy
        float[] cumulativeEnergy = new float[numTiers];
        if (weights != null) {
            float[] allPhi = TransformOperator.computeCumulativeEnergy(weights, dimension, loraDim);
            for (int k = 0; k < numTiers; k++) {
                cumulativeEnergy[k] = allPhi[tiers[k] - 1];
            }
        } else {
            // Equal distribution fallback
            for (int k = 0; k < numTiers; k++) {
                cumulativeEnergy[k] = (float) tiers[k] / dimension;
            }
        }

        return new FlatIndex(mappedBase, idsSegment, tierSegments, metadataSegment,
                planetId, planetRadius, dimension, numTiers, tiers, totalRecords, cumulativeEnergy);
    }

    @Override
    public void insert(VectorRecord record) {
        throw new UnsupportedOperationException("Insert is not supported on read-only memory-mapped Index.");
    }

    @Override
    public List<SearchResult> search(float[] query, int k) {
        List<SearchResult>[] results = batchSearch(new float[][]{query}, k);
        return results[0];
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<SearchResult>[] batchSearch(float[][] queries, int k) {
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

            // Sort merged matches by distance (ascending)
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

    private void executeKnnRange(long startIdx, long endIdx, float[][] queries, int k, long[][] myIds, int[][] myDists) {
        long idsAddr = idsSegment.address();
        long metadataAddr = metadataSegment.address();
        int numQueries = queries.length;

        // Binarize all queries once
        long[][] bQueries = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            bQueries[q] = transformOperator.transformAndQuantize(queries[q]);
        }

        // Determine target active truncation tier T
        int T = 0;
        for (int i = 0; i < numTiers; i++) {
            if (cumulativeEnergy[i] >= targetEnergyBudget) {
                T = i;
                break;
            }
        }

        long[] tierAddrs = new long[numTiers];
        for (int i = 0; i < numTiers; i++) {
            tierAddrs[i] = tierSegments[i].address();
        }

        int[] tierLongs = new int[numTiers];
        int prevBound = 0;
        for (int i = 0; i < numTiers; i++) {
            tierLongs[i] = (tiers[i] - prevBound) / 64;
            prevBound = tiers[i];
        }

        int totalLongs = dimension / 64;
        long[] dbWords = new long[totalLongs];

        for (long i = startIdx; i < endIdx; i++) {
            // Gate 1: Tombstone & Attribute Mask
            long metaVal = UNSAFE.getLong(metadataAddr + (i * 8));
            if ((metaVal & 1L) == 1L) {
                continue; // Tombstone active -> Deleted
            }

            // Gate 2: QEG (Quantization Entropy Gating)
            long t0Val = UNSAFE.getLong(tierAddrs[0] + (i * 8));
            if ((t0Val & (1L << 63)) == 0L) {
                continue; // Early exit -> Flat background terrain
            }

            long recordId = UNSAFE.getLong(idsAddr + (i * 8));

            // Load all words of record i locally to avoid redundant off-heap reads
            dbWords[0] = t0Val;
            int wordIdx = 1;
            for (int tierIdx = 1; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                long baseOffset = i * (numLongs * 8L);
                long tAddr = tierAddrs[tierIdx] + baseOffset;
                for (int l = 0; l < numLongs; l++) {
                    dbWords[wordIdx++] = UNSAFE.getLong(tAddr + (l * 8));
                }
            }

            // Gate 3: XOR-Popcount Cascade
            for (int q = 0; q < numQueries; q++) {
                int[] dists = myDists[q];
                int currentLimit = dists[k - 1];

                int totalDist = 0;
                int queryOffsetLongs = 0;

                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int tierDist = 0;
                    for (int l = 0; l < numLongs; l++) {
                        long qWord = bQueries[q][queryOffsetLongs + l];
                        long dbWord = dbWords[queryOffsetLongs + l];
                        tierDist += Long.bitCount(qWord ^ dbWord);
                    }
                    totalDist += tierDist;
                    queryOffsetLongs += numLongs;

                    // If dynamic threshold is exceeded, break early
                    if (totalDist > currentLimit) {
                        break;
                    }
                }

                if (totalDist < currentLimit) {
                    long[] ids = myIds[q];
                    int pos = k - 1;
                    while (pos > 0 && totalDist < dists[pos - 1]) {
                        dists[pos] = dists[pos - 1];
                        ids[pos] = ids[pos - 1];
                        pos--;
                    }
                    dists[pos] = totalDist;
                    ids[pos] = recordId;
                }
            }
        }
    }

    @Override
    public long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        if (queries == null || queries.length == 0) return 0;
        int numQueries = queries.length;
        long numRecords = this.size;
        if (numRecords == 0) return 0;

        long currentChunkSize = this.chunkSize;
        long numChunks = (numRecords + currentChunkSize - 1) / currentChunkSize;
        CountDownLatch latch = new CountDownLatch((int) numChunks);

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
                // Check if resonance threshold is reached (e.g. >= 5 votes)
                if (Integer.bitCount(mergedVal & 0xFF) >= 5) {
                    resonantCount++;
                }
            }
            return resonantCount;
        }).sum();
    }

    private void executeVotingRange(long startIdx, long endIdx, float[][] queries, int[] families, int[] thresholds, MemorySegment localMask) {
        long metadataAddr = metadataSegment.address();
        long localMaskAddr = localMask.address();
        int numQueries = queries.length;

        // Binarize all queries once
        long[][] bQueries = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            bQueries[q] = transformOperator.transformAndQuantize(queries[q]);
        }

        // Determine target active truncation tier T
        int T = 0;
        for (int i = 0; i < numTiers; i++) {
            if (cumulativeEnergy[i] >= targetEnergyBudget) {
                T = i;
                break;
            }
        }

        long[] tierAddrs = new long[numTiers];
        for (int i = 0; i < numTiers; i++) {
            tierAddrs[i] = tierSegments[i].address();
        }

        int[] tierLongs = new int[numTiers];
        int prevBound = 0;
        for (int i = 0; i < numTiers; i++) {
            tierLongs[i] = (tiers[i] - prevBound) / 64;
            prevBound = tiers[i];
        }

        int totalLongs = dimension / 64;
        long[] dbWords = new long[totalLongs];

        for (long i = startIdx; i < endIdx; i++) {
            // Gate 1: Tombstone & Attribute Mask
            long metaVal = UNSAFE.getLong(metadataAddr + (i * 8));
            if ((metaVal & 1L) == 1L) {
                continue; // Tombstone active
            }

            // Gate 2: QEG
            long t0Val = UNSAFE.getLong(tierAddrs[0] + (i * 8));
            if ((t0Val & (1L << 63)) == 0L) {
                continue; // Early exit
            }

            // Load all words of record i locally to avoid redundant off-heap reads
            dbWords[0] = t0Val;
            int wordIdx = 1;
            for (int tierIdx = 1; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                long baseOffset = i * (numLongs * 8L);
                long tAddr = tierAddrs[tierIdx] + baseOffset;
                for (int l = 0; l < numLongs; l++) {
                    dbWords[wordIdx++] = UNSAFE.getLong(tAddr + (l * 8));
                }
            }

            byte maskVal = 0;
            for (int q = 0; q < numQueries; q++) {
                int totalDist = 0;
                int queryOffsetLongs = 0;
                boolean earlyExit = false;

                for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                    int numLongs = tierLongs[tierIdx];
                    int tierDist = 0;
                    for (int l = 0; l < numLongs; l++) {
                        long qWord = bQueries[q][queryOffsetLongs + l];
                        long dbWord = dbWords[queryOffsetLongs + l];
                        tierDist += Long.bitCount(qWord ^ dbWord);
                    }
                    totalDist += tierDist;
                    queryOffsetLongs += numLongs;

                    if (totalDist > thresholds[q]) {
                        earlyExit = true;
                        break;
                    }
                }

                if (!earlyExit) {
                    maskVal |= (byte) (1 << families[q]);
                }
            }
            UNSAFE.putByte(localMaskAddr + i, maskVal);
        }
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public byte getPlanetId() {
        return planetId;
    }

    @Override
    public long getPlanetRadius() {
        return planetRadius;
    }

    @Override
    public int getTierCount() {
        return numTiers;
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
