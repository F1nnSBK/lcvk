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
import java.util.Set;
import java.util.HashSet;
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
    private final MemorySegment fp16Segment; // Optional FP16 sidecar; null if not present

    private final byte planetId;
    private final long planetRadius;
    private final int dimension;
    private final int numTiers;
    private final int[] tiers;
    private final long size;

    private final TransformOperator transformOperator;
    private final float[] cumulativeEnergy;
    private double targetEnergyBudget = 0.90; // Default energy threshold tau
    private final int qMode;
    private final int[] tierLongs;
    private final int[] tierOffsets;

    /** Dimension threshold below which we skip Hamming compression and use direct float L2. */
    private static final int SIMD_FLOAT_DIM_THRESHOLD = 32;

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
                    index.executeVotingRange(event.startIdx, event.endIdx, event.queries, event.families,
                            event.thresholds, event.threadLocalMasks[threadId]);
                } else {
                    index.executeKnnRange(event.startIdx, event.endIdx, event.queries, event.k,
                            event.threadLocalIds[threadId], event.threadLocalDists[threadId]);
                }
            } finally {
                event.latch.countDown();
            }
        }
    }

    public FlatIndex(MemorySegment baseSegment, MemorySegment idsSegment, MemorySegment[] tierSegments,
            MemorySegment metadataSegment, MemorySegment fp16Segment,
            byte planetId, long planetRadius, int dimension, int numTiers, int[] tiers, long size,
            float[] cumulativeEnergy, int qMode) {
        this.baseSegment = baseSegment;
        this.idsSegment = idsSegment;
        this.tierSegments = tierSegments;
        this.metadataSegment = metadataSegment;
        this.fp16Segment = fp16Segment;
        this.planetId = planetId;
        this.planetRadius = planetRadius;
        this.dimension = dimension;
        this.numTiers = numTiers;
        this.tiers = tiers;
        this.size = size;
        this.cumulativeEnergy = cumulativeEnergy;
        this.qMode = qMode;
        this.tierLongs = new int[numTiers];
        int prevBoundVal = 0;
        for (int idx = 0; idx < numTiers; idx++) {
            this.tierLongs[idx] = (tiers[idx] - prevBoundVal) / 64;
            prevBoundVal = tiers[idx];
        }

        this.tierOffsets = new int[numTiers];
        int offset = 0;
        for (int idx = 0; idx < numTiers; idx++) {
            this.tierOffsets[idx] = offset;
            offset += this.tierLongs[idx];
        }

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
                new BlockingWaitStrategy());
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

    public long getTierAddress(int tierIdx) {
        if (tierIdx < 0 || tierIdx >= numTiers) {
            return 0;
        }
        return tierSegments[tierIdx].address();
    }

    public long getTierByteSize(int tierIdx) {
        if (tierIdx < 0 || tierIdx >= numTiers) {
            return 0;
        }
        return tierSegments[tierIdx].byteSize();
    }

    public long getMetadataAddress() {
        return metadataSegment.address();
    }

    public long getMetadataByteSize() {
        return metadataSegment.byteSize();
    }

    public long getIdsAddress() {
        return idsSegment.address();
    }

    public long getIdsByteSize() {
        return idsSegment.byteSize();
    }

    public TransformOperator getTransformOperator() {
        return transformOperator;
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

        byte qModeByte = mappedBase.get(ValueLayout.JAVA_BYTE, 61);
        int qMode = qModeByte & 0xFF;

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
            long bytesPerRecord = switch (qMode) {
                case 1 -> (width / 4);   // 2-bit ternary
                case 2 -> (width * 4L);  // Float-Hybrid: raw float32
                default -> (width / 8);  // 1-bit sign
            };
            try (FileChannel channel = FileChannel.open(tierPath, StandardOpenOption.READ)) {
                tierSegments[k] = channel.map(FileChannel.MapMode.READ_ONLY, 0, totalRecords * bytesPerRecord,
                        Arena.global());
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

        // 6. Optionally map FP16 sidecar (basePath + "_fp16.bin")
        MemorySegment fp16Segment = null;
        Path fp16Path = Path.of(basePath + "_fp16.bin");
        if (fp16Path.toFile().exists()) {
            try (FileChannel channel = FileChannel.open(fp16Path, StandardOpenOption.READ)) {
                fp16Segment = channel.map(FileChannel.MapMode.READ_ONLY, 0,
                        totalRecords * dimension * 2L, Arena.global());
            }
        }

        return new FlatIndex(mappedBase, idsSegment, tierSegments, metadataSegment, fp16Segment,
                planetId, planetRadius, dimension, numTiers, tiers, totalRecords, cumulativeEnergy, qMode);
    }

    @Override
    public void insert(VectorRecord record) {
        throw new UnsupportedOperationException("Insert is not supported on read-only memory-mapped Index.");
    }

    @Override
    public List<SearchResult> search(float[] query, int k) {
        List<SearchResult>[] results = batchSearch(new float[][] { query }, k);
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

        // Stage 1: Coarse Hamming/Ternary search returning candidates as internal row indices.
        // We evaluate more candidates for reranking.
        int kCandidate = (int) Math.min(numRecords, Math.max(100, 3 * k));

        // Allocate thread-local structures to hold top-K candidates
        long[][][] threadLocalIds = new long[numWorkers][numQueries][kCandidate];
        int[][][] threadLocalDists = new int[numWorkers][numQueries][kCandidate];

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
                event.setKnn(startIdx, endIdx, queries, kCandidate, threadLocalIds, threadLocalDists, latch);
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

        // Precondition and rotate queries to the same space as quantized vectors
        float[][] zQueries = new float[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            zQueries[q] = transformOperator.preconditionAndRotate(queries[q]);
        }

        long idsAddr = idsSegment.address();

        // Stage 2: Asymmetric Reranking (parallelized over queries to minimize latency)
        List<SearchResult>[] finalResults = new List[numQueries];
        IntStream.range(0, numQueries).forEach(q -> {
            List<SearchResult> merged = new ArrayList<>();
            for (int w = 0; w < numWorkers; w++) {
                long[] ids = threadLocalIds[w][q];
                int[] dists = threadLocalDists[w][q];
                for (int i = 0; i < kCandidate; i++) {
                    if (dists[i] != Integer.MAX_VALUE) {
                        merged.add(new SearchResult(ids[i], dists[i]));
                    }
                }
            }

            // Sort merged matches by Stage 1 distance (ascending)
            merged.sort((r1, r2) -> {
                int cmp = Integer.compare(r1.score(), r2.score());
                if (cmp != 0)
                    return cmp;
                return Long.compare(r1.id(), r2.id());
            });

            // Dedup candidates and keep the top kCandidate unique candidates
            List<Long> candidates = new ArrayList<>();
            Set<Long> seen = new HashSet<>();
            for (SearchResult r : merged) {
                long rowIdx = r.id();
                if (seen.add(rowIdx)) {
                    candidates.add(rowIdx);
                    if (candidates.size() >= kCandidate) {
                        break;
                    }
                }
            }

            // Compute exact query L2 norm and sum for the distance offset
            double queryL2Norm = 0.0;
            double querySum = 0.0;
            for (float val : zQueries[q]) {
                queryL2Norm += val * val;
                querySum += val;
            }

            // For QMode 2 (Float-Hybrid), Stage 1 already produced exact L2 distances.
            // Skip asymmetric reranking; directly resolve record IDs.
            if (qMode == 2) {
                List<SearchResult> queryResults = new ArrayList<>();
                int limit = Math.min(k, candidates.size());
                for (int i = 0; i < limit; i++) {
                    long rowIdx = candidates.get(i);
                    long recordId = UNSAFE.getLong(idsAddr + (rowIdx * 8));
                    queryResults.add(new SearchResult(recordId, 0));
                }
                finalResults[q] = queryResults;
                return;
            }

            // Rerank candidates using exact asymmetric float-ternary L2 distance
            class RerankedCandidate {
                final long rowIdx;
                final double distance;
                RerankedCandidate(long rowIdx, double distance) {
                    this.rowIdx = rowIdx;
                    this.distance = distance;
                }
            }

            // Choose reranking strategy:
            //  - If FP16 sidecar is present: use exact L2 on original (pre-rotation) vectors → highest recall
            //  - Otherwise: use asymmetric binary/ternary L2 estimator (cheaper, slightly lower recall)
            List<RerankedCandidate> reranked = new ArrayList<>();
            if (fp16Segment != null) {
                float[] rawQuery = queries[q]; // raw (pre-rotation) query
                for (long rowIdx : candidates) {
                    double dist = computeExactL2FP16(rawQuery, rowIdx);
                    reranked.add(new RerankedCandidate(rowIdx, dist));
                }
            } else {
                for (long rowIdx : candidates) {
                    double dist = computeAsymmetricL2DistanceOffHeap(zQueries[q], queryL2Norm, querySum, rowIdx);
                    reranked.add(new RerankedCandidate(rowIdx, dist));
                }
            }

            // Sort by distance (ascending)
            reranked.sort((c1, c2) -> Double.compare(c1.distance, c2.distance));

            // Select top K and resolve original record IDs
            List<SearchResult> queryResults = new ArrayList<>();
            int limit = Math.min(k, reranked.size());
            for (int i = 0; i < limit; i++) {
                RerankedCandidate c = reranked.get(i);
                long recordId = UNSAFE.getLong(idsAddr + (c.rowIdx * 8));
                queryResults.add(new SearchResult(recordId, (int) (c.distance * 1000000.0)));
            }
            finalResults[q] = queryResults;
        });

        return finalResults;
    }

    private void executeKnnRange(long startIdx, long endIdx, float[][] queries, int k, long[][] myIds,
            int[][] myDists) {
        long metadataAddr = metadataSegment.address();
        int numQueries = queries.length;

        // Binarize all queries once
        long[][] bQueries = new long[numQueries][];
        long[][] bQueriesMask = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            if (qMode == 1) { // 2-bit mode
                float[] z = transformOperator.preconditionAndRotate(queries[q]);
                float qThreshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(z, qThreshold);
                bQueries[q] = packed[0];
                bQueriesMask[q] = packed[1];
            } else {
                bQueries[q] = transformOperator.transformAndQuantize(queries[q]);
            }
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

        // --- SIMD Float-L2 dispatch for D <= SIMD_FLOAT_DIM_THRESHOLD (1-bit mode only) ---
        if (dimension <= SIMD_FLOAT_DIM_THRESHOLD && qMode == 0) {
            // For tiny dimensions the Hamming approximation introduces large relative error.
            // Instead: reconstruct ±1 float vectors from sign bits and use exact VectorAPI L2.
            float[][] rotQueries = new float[numQueries][];
            for (int q = 0; q < numQueries; q++) {
                rotQueries[q] = transformOperator.preconditionAndRotate(queries[q]);
            }
            long tier0Addr = tierAddrs[0];
            int numLongs0 = tierLongs[0];
            float[] dbFloat = new float[dimension];
            for (long i = startIdx; i < endIdx; i++) {
                long metaVal = UNSAFE.getLong(metadataAddr + (i * 8));
                if ((metaVal & 1L) == 1L) continue;
                long baseOff = i * (numLongs0 * 8L);
                for (int d = 0; d < dimension; d++) {
                    long word = UNSAFE.getLong(tier0Addr + baseOff + (d / 64) * 8);
                    dbFloat[d] = ((word >>> (d % 64)) & 1L) != 0L ? 1.0f : -1.0f;
                }
                for (int q = 0; q < numQueries; q++) {
                    float dist = transformOperator.computeL2Float(rotQueries[q], dbFloat);
                    int iDist = (int) (dist * 1000f);
                    int[] dists = myDists[q];
                    long[] ids = myIds[q];
                    if (iDist < dists[k - 1]) {
                        int pos = k - 1;
                        while (pos > 0 && iDist < dists[pos - 1]) {
                            dists[pos] = dists[pos - 1];
                            ids[pos] = ids[pos - 1];
                            pos--;
                        }
                        dists[pos] = iDist;
                        ids[pos] = i;
                    }
                }
            }
            return; // Skip standard Hamming path
        }
        // --- End SIMD Float-L2 dispatch ---

        // --- QMode 2: Float-Hybrid raw bypass ---
        if (qMode == 2) {
            // Tier files store raw rotated float32 values (no quantization).
            // Read them directly off-heap and compute exact VectorAPI L2.
            float[][] rotQueries = new float[numQueries][];
            for (int q = 0; q < numQueries; q++) {
                rotQueries[q] = transformOperator.preconditionAndRotate(queries[q]);
            }
            float[] dbFloat = new float[dimension];
            int dimOffset = 0;
            for (long i = startIdx; i < endIdx; i++) {
                long metaVal = UNSAFE.getLong(metadataAddr + (i * 8));
                if ((metaVal & 1L) == 1L) continue;
                // Reconstruct full float vector from all tiers
                dimOffset = 0;
                for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                    int startDim = tierIdx == 0 ? 0 : tiers[tierIdx - 1];
                    int width = tiers[tierIdx] - startDim;
                    long tAddr = tierAddrs[tierIdx] + i * (width * 4L);
                    for (int d = 0; d < width; d++) {
                        dbFloat[dimOffset + d] = Float.intBitsToFloat(UNSAFE.getInt(tAddr + d * 4L));
                    }
                    dimOffset += width;
                }
                for (int q = 0; q < numQueries; q++) {
                    float dist = transformOperator.computeL2Float(rotQueries[q], dbFloat);
                    int iDist = (int) (dist * 1000f);
                    int[] dists = myDists[q];
                    long[] ids = myIds[q];
                    if (iDist < dists[k - 1]) {
                        int pos = k - 1;
                        while (pos > 0 && iDist < dists[pos - 1]) {
                            dists[pos] = dists[pos - 1];
                            ids[pos] = ids[pos - 1];
                            pos--;
                        }
                        dists[pos] = iDist;
                        ids[pos] = i;
                    }
                }
            }
            return;
        }
        // --- End QMode 2 dispatch ---

        int totalLongs = dimension / 64;
        long[] dbWords = new long[totalLongs];
        long[] dbMasks = new long[totalLongs];

        for (long i = startIdx; i < endIdx; i++) {
            // Gate 1: Tombstone & Attribute Mask
            long metaVal = UNSAFE.getLong(metadataAddr + (i * 8));
            if ((metaVal & 1L) == 1L) {
                continue; // Tombstone active -> Deleted
            }

            int loadedTiers = 1; // Bit 0 is loaded
            if (qMode == 1) { // 2-bit mode
                int numLongs0 = tierLongs[0];
                long baseOffset0 = i * (numLongs0 * 16L);
                long tAddr0 = tierAddrs[0] + baseOffset0;
                for (int l = 0; l < numLongs0; l++) {
                    dbWords[l] = UNSAFE.getLong(tAddr0 + (l * 8));
                    dbMasks[l] = UNSAFE.getLong(tAddr0 + (numLongs0 * 8L) + (l * 8));
                }
            } else { // 1-bit mode
                int numLongs0 = tierLongs[0];
                long baseOffset0 = i * (numLongs0 * 8L);
                long tAddr0 = tierAddrs[0] + baseOffset0;
                for (int l = 0; l < numLongs0; l++) {
                    dbWords[l] = UNSAFE.getLong(tAddr0 + (l * 8));
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

                    // Load tierIdx on demand if not loaded yet
                    if ((loadedTiers & (1 << tierIdx)) == 0) {
                        int offset = tierOffsets[tierIdx];
                        if (qMode == 1) {
                            long baseOffset = i * (numLongs * 16L);
                            long tAddr = tierAddrs[tierIdx] + baseOffset;
                            for (int l = 0; l < numLongs; l++) {
                                dbWords[offset + l] = UNSAFE.getLong(tAddr + (l * 8));
                                dbMasks[offset + l] = UNSAFE.getLong(tAddr + (numLongs * 8L) + (l * 8));
                            }
                        } else {
                            long baseOffset = i * (numLongs * 8L);
                            long tAddr = tierAddrs[tierIdx] + baseOffset;
                            for (int l = 0; l < numLongs; l++) {
                                dbWords[offset + l] = UNSAFE.getLong(tAddr + (l * 8));
                            }
                        }
                        loadedTiers |= (1 << tierIdx);
                    }

                    int tierDist = 0;
                    if (qMode == 1) { // 2-bit mode
                        for (int l = 0; l < numLongs; l++) {
                            long qSign = bQueries[q][queryOffsetLongs + l];
                            long qMask = bQueriesMask[q][queryOffsetLongs + l];
                            long dbSign = dbWords[queryOffsetLongs + l];
                            long dbMask = dbMasks[queryOffsetLongs + l];
                            
                            long mask4 = dbMask & qMask & (dbSign ^ qSign);
                            long mask1 = dbMask ^ qMask;
                            tierDist += 4 * Long.bitCount(mask4) + Long.bitCount(mask1);
                        }
                    } else { // 1-bit mode
                        for (int l = 0; l < numLongs; l++) {
                            long qWord = bQueries[q][queryOffsetLongs + l];
                            long dbWord = dbWords[queryOffsetLongs + l];
                            tierDist += Long.bitCount(qWord ^ dbWord);
                        }
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
                    ids[pos] = i; // Save the internal row index
                }
            }
        }
    }

    private double computeAsymmetricL2DistanceOffHeap(float[] query, double queryL2Norm, double querySum, long rowIdx) {
        int totalMaskPopcount = 0;
        int queryOffsetLongs = 0;
        
        if (qMode == 1) { // 2-bit mode
            double sumPositive = 0.0;
            double sumActive = 0.0;
            for (int tierIdx = 0; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                long baseOffset = rowIdx * (numLongs * 16L);
                long tAddr = tierSegments[tierIdx].address() + baseOffset;
                
                for (int l = 0; l < numLongs; l++) {
                    long mask = UNSAFE.getLong(tAddr + (numLongs * 8L) + (l * 8));
                    if (mask == 0L) {
                        continue;
                    }
                    totalMaskPopcount += Long.bitCount(mask);
                    long word = UNSAFE.getLong(tAddr + (l * 8));
                    
                    int bitOffset = (queryOffsetLongs + l) * 64;
                    int limit = Math.min(64, query.length - bitOffset);
                    long limitMask = limit == 64 ? -1L : (1L << limit) - 1L;
                    long active = mask & limitMask;
                    while (active != 0) {
                        int bitIdx = Long.numberOfTrailingZeros(active);
                        float qVal = query[bitOffset + bitIdx];
                        sumActive += qVal;
                        if (((word >>> bitIdx) & 1L) != 0L) {
                            sumPositive += qVal;
                        }
                        active &= active - 1;
                    }
                }
                queryOffsetLongs += numLongs;
            }
            return totalMaskPopcount + queryL2Norm - 4.0 * sumPositive + 2.0 * sumActive;
        } else { // 1-bit mode
            double sumPositive = 0.0;
            // Tier 0
            long t0Addr = tierSegments[0].address() + (rowIdx * 8);
            long word0 = UNSAFE.getLong(t0Addr);
            int limit0 = Math.min(64, query.length);
            long limitMask0 = limit0 == 64 ? -1L : (1L << limit0) - 1L;
            long active0 = word0 & limitMask0;
            while (active0 != 0) {
                int bitIdx = Long.numberOfTrailingZeros(active0);
                sumPositive += query[bitIdx];
                active0 &= active0 - 1;
            }
            
            queryOffsetLongs = 1;
            for (int tierIdx = 1; tierIdx < numTiers; tierIdx++) {
                int numLongs = tierLongs[tierIdx];
                long baseOffset = rowIdx * (numLongs * 8L);
                long tAddr = tierSegments[tierIdx].address() + baseOffset;
                
                for (int l = 0; l < numLongs; l++) {
                    long word = UNSAFE.getLong(tAddr + (l * 8));
                    int bitOffset = (queryOffsetLongs + l) * 64;
                    int limit = Math.min(64, query.length - bitOffset);
                    long limitMask = limit == 64 ? -1L : (1L << limit) - 1L;
                    long active = word & limitMask;
                    while (active != 0) {
                        int bitIdx = Long.numberOfTrailingZeros(active);
                        sumPositive += query[bitOffset + bitIdx];
                        active &= active - 1;
                    }
                }
                queryOffsetLongs += numLongs;
            }
            return query.length + queryL2Norm + 2.0 * querySum - 4.0 * sumPositive;
        }
    }

    /**
     * Computes exact L2 distance between a raw (pre-rotation) query and the stored FP16 vector at rowIdx.
     * Requires fp16Segment to be non-null.
     * The query must be in the original (non-rotated) float space.
     *
     * @param rawQuery raw float query vector (pre-rotation)
     * @param rowIdx   row index in the FP16 sidecar
     * @return exact L2 squared distance (float precision)
     */
    private double computeExactL2FP16(float[] rawQuery, long rowIdx) {
        long fp16Addr = fp16Segment.address();
        long rowOffset = rowIdx * dimension * 2L;
        double sum = 0.0;
        for (int d = 0; d < dimension; d++) {
            short fp16 = UNSAFE.getShort(fp16Addr + rowOffset + d * 2L);
            float dbVal = Float.float16ToFloat(fp16);
            double diff = rawQuery[d] - dbVal;
            sum += diff * diff;
        }
        return sum;
    }

    @Override
    public long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        if (queries == null || queries.length == 0)
            return 0;
        int numQueries = queries.length;
        long numRecords = this.size;
        if (numRecords == 0)
            return 0;

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

    private void executeVotingRange(long startIdx, long endIdx, float[][] queries, int[] families, int[] thresholds,
            MemorySegment localMask) {
        long metadataAddr = metadataSegment.address();
        long localMaskAddr = localMask.address();
        int numQueries = queries.length;

        // Binarize all queries once
        long[][] bQueries = new long[numQueries][];
        long[][] bQueriesMask = new long[numQueries][];
        for (int q = 0; q < numQueries; q++) {
            if (qMode == 1) { // 2-bit mode
                float[] z = transformOperator.preconditionAndRotate(queries[q]);
                float qThreshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                long[][] packed = transformOperator.quantize2Bit(z, qThreshold);
                bQueries[q] = packed[0];
                bQueriesMask[q] = packed[1];
            } else {
                bQueries[q] = transformOperator.transformAndQuantize(queries[q]);
            }
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



        int totalLongs = dimension / 64;
        long[] dbWords = new long[totalLongs];
        long[] dbMasks = new long[totalLongs];
        boolean[] active = new boolean[numQueries];
        int[] accumDists = new int[numQueries];

        for (long i = startIdx; i < endIdx; i++) {
            // Gate 1: Tombstone & Attribute Mask
            long metaVal = UNSAFE.getLong(metadataAddr + (i * 8));
            if ((metaVal & 1L) == 1L) {
                continue; // Tombstone active
            }

            int loadedTiers = 0;
            if (qMode == 1) { // 2-bit mode
                // Gate 2: QEG
                int numLongs0 = tierLongs[0];
                long baseOffset0 = i * (numLongs0 * 16L);
                long tAddr0 = tierAddrs[0] + baseOffset0;
                long t0Sign = UNSAFE.getLong(tAddr0);
                long t0Mask = UNSAFE.getLong(tAddr0 + (numLongs0 * 8L));
                if ((t0Mask & (1L << 63)) == 0L || (t0Sign & (1L << 63)) == 0L) {
                    continue; // Early exit
                }
                dbWords[0] = t0Sign;
                dbMasks[0] = t0Mask;
                for (int l = 1; l < numLongs0; l++) {
                    dbWords[l] = UNSAFE.getLong(tAddr0 + (l * 8));
                    dbMasks[l] = UNSAFE.getLong(tAddr0 + (numLongs0 * 8L) + (l * 8));
                }
                loadedTiers = 1;
            } else { // 1-bit mode
                // Gate 2: QEG
                int numLongs0 = tierLongs[0];
                long baseOffset0 = i * (numLongs0 * 8L);
                long tAddr0 = tierAddrs[0] + baseOffset0;
                long t0Val = UNSAFE.getLong(tAddr0);
                if ((t0Val & (1L << 63)) == 0L) {
                    continue; // Early exit
                }
                dbWords[0] = t0Val;
                for (int l = 1; l < numLongs0; l++) {
                    dbWords[l] = UNSAFE.getLong(tAddr0 + (l * 8));
                }
                loadedTiers = 1;
            }

            int activeCount = numQueries;
            for (int q = 0; q < numQueries; q++) {
                active[q] = true;
                accumDists[q] = 0;
            }

            for (int tierIdx = 0; tierIdx <= T; tierIdx++) {
                if (activeCount == 0) {
                    break;
                }
                int numLongs = tierLongs[tierIdx];
                int offset = tierOffsets[tierIdx];

                // Load tierIdx on demand if not loaded yet
                if ((loadedTiers & (1 << tierIdx)) == 0) {
                    if (qMode == 1) {
                        long baseOffset = i * (numLongs * 16L);
                        long tAddr = tierAddrs[tierIdx] + baseOffset;
                        for (int l = 0; l < numLongs; l++) {
                            dbWords[offset + l] = UNSAFE.getLong(tAddr + (l * 8));
                            dbMasks[offset + l] = UNSAFE.getLong(tAddr + (numLongs * 8L) + (l * 8));
                        }
                    } else {
                        long baseOffset = i * (numLongs * 8L);
                        long tAddr = tierAddrs[tierIdx] + baseOffset;
                        for (int l = 0; l < numLongs; l++) {
                            dbWords[offset + l] = UNSAFE.getLong(tAddr + (l * 8));
                        }
                    }
                    loadedTiers |= (1 << tierIdx);
                }

                // Compute distance for all active queries
                for (int q = 0; q < numQueries; q++) {
                    if (!active[q]) continue;

                    int tierDist = 0;
                    if (qMode == 1) { // 2-bit mode
                        for (int l = 0; l < numLongs; l++) {
                            long qSign = bQueries[q][offset + l];
                            long qMask = bQueriesMask[q][offset + l];
                            long dbSign = dbWords[offset + l];
                            long dbMask = dbMasks[offset + l];
                            
                            long mask4 = dbMask & qMask & (dbSign ^ qSign);
                            long mask1 = dbMask ^ qMask;
                            tierDist += 4 * Long.bitCount(mask4) + Long.bitCount(mask1);
                        }
                    } else { // 1-bit mode
                        for (int l = 0; l < numLongs; l++) {
                            long qWord = bQueries[q][offset + l];
                            long dbWord = dbWords[offset + l];
                            tierDist += Long.bitCount(qWord ^ dbWord);
                        }
                    }
                    accumDists[q] += tierDist;
                    if (accumDists[q] > thresholds[q]) {
                        active[q] = false;
                        activeCount--;
                    }
                }
            }

            byte maskVal = 0;
            if (activeCount > 0) {
                for (int q = 0; q < numQueries; q++) {
                    if (active[q]) {
                        maskVal |= (byte) (1 << families[q]);
                    }
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
