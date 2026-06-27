package org.pithos;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Coordinate layer managing active multi-tier Index instances and compilation.
 */
public class VectorDb {
    private final Map<String, Index> indices;
    /** Per-index optional write buffers (LSM delta layer). */
    private final Map<String, DeltaBuffer> deltaBuffers;

    public VectorDb() {
        this.indices = new ConcurrentHashMap<>();
        this.deltaBuffers = new ConcurrentHashMap<>();
    }

    /**
     * Maps a multi-tier index files off-heap and registers it.
     */
    public Index loadIndex(String name, String basePath, float[] weights, int loraDim) throws IOException {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Index name cannot be empty");
        }
        Index index = FlatIndex.mapFile(basePath, weights, loraDim);
        indices.put(name, index);
        return index;
    }

    public Index getIndex(String name) {
        return indices.get(name);
    }

    public boolean dropIndex(String name) {
        deltaBuffers.remove(name); // also drop associated delta buffer
        Index index = indices.remove(name);
        if (index != null) {
            try {
                index.close();
            } catch (Exception e) {
                // ignore
            }
        }
        return index != null;
    }

    public void close() {
        for (Index index : indices.values()) {
            try {
                index.close();
            } catch (Exception e) {
                // ignore
            }
        }
        indices.clear();
        deltaBuffers.clear();
    }

    // -------------------------------------------------------------------------
    // Delta Buffer API (LSM layer)
    // -------------------------------------------------------------------------

    /**
     * Creates an in-memory delta buffer for the given index.
     * The buffer enables real-time inserts without modifying the immutable base
     * index.
     *
     * @param indexName      name of the base index
     * @param flushThreshold soft limit on live entries before flush is recommended
     * @return the new DeltaBuffer
     * @throws IllegalArgumentException if the index does not exist
     */
    public DeltaBuffer createDeltaBuffer(String indexName, int flushThreshold) {
        Index index = indices.get(indexName);
        if (index == null)
            throw new IllegalArgumentException("Unknown index: " + indexName);
        DeltaBuffer buf = new DeltaBuffer(index.getDimension(), flushThreshold);
        deltaBuffers.put(indexName, buf);
        return buf;
    }

    /** Returns the delta buffer for the given index, or null if none exists. */
    public DeltaBuffer getDeltaBuffer(String indexName) {
        return deltaBuffers.get(indexName);
    }

    /**
     * Inserts a vector into the delta buffer for the given index.
     *
     * @return true on success, false if no delta buffer is registered
     */
    public boolean insertIntoDelta(String indexName, long id, float[] vector) {
        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null)
            return false;
        buf.insert(id, vector);
        return true;
    }

    /**
     * Soft-deletes a record from the delta buffer (tombstone).
     *
     * @return true if at least one entry was tombstoned
     */
    public boolean deleteFromDelta(String indexName, long id) {
        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null)
            return false;
        return buf.delete(id);
    }

    /**
     * Backs up the live entries of the delta buffer to a binary file.
     *
     * @param indexName name of the index whose delta buffer to back up
     * @param path      target file path
     * @throws IOException           on I/O failure
     * @throws IllegalStateException if no delta buffer exists for the index
     */
    public void backupDelta(String indexName, String path) throws IOException {
        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null)
            throw new IllegalStateException("No delta buffer for index: " + indexName);
        buf.serializeToPath(path);
    }

    /**
     * Restores a delta buffer from a previously backed-up binary file.
     * Replaces any existing delta buffer for the index.
     *
     * @param indexName      name of the index
     * @param path           path to the backup file
     * @param flushThreshold flush threshold for the restored buffer
     * @throws IOException on I/O failure
     */
    public void restoreDelta(String indexName, String path, int flushThreshold) throws IOException {
        DeltaBuffer buf = DeltaBuffer.deserializeFromPath(path, flushThreshold);
        deltaBuffers.put(indexName, buf);
    }

    /**
     * Performs a unified search that queries both the base index and the delta
     * buffer,
     * merges the results, and returns the top-K by score.
     *
     * @param indexName name of the index
     * @param query     raw float query vector
     * @param k         number of results
     * @return merged top-K results
     */
    public List<Index.SearchResult> searchMerged(String indexName, float[] query, int k) {
        Index index = indices.get(indexName);
        if (index == null)
            throw new IllegalArgumentException("Unknown index: " + indexName);

        List<Index.SearchResult> baseResults = index.search(query, k);

        DeltaBuffer buf = deltaBuffers.get(indexName);
        if (buf == null || buf.liveSize() == 0) {
            return baseResults;
        }

        List<Index.SearchResult> deltaResults = buf.searchKnn(query, k);

        // Merge and deduplicate by ID, then take top-K by score
        java.util.Map<Long, Index.SearchResult> merged = new java.util.LinkedHashMap<>();
        for (Index.SearchResult r : baseResults)
            merged.put(r.id(), r);
        for (Index.SearchResult r : deltaResults)
            merged.putIfAbsent(r.id(), r);

        return merged.values().stream()
                .sorted((a, b) -> Integer.compare(a.score(), b.score()))
                .limit(k)
                .toList();
    }

    /**
     * Compiles raw float records into a multi-tier, cache-aligned database file
     * layout.
     */
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, 0);
    }

    /**
     * Compiles raw float records into a multi-tier, cache-aligned database file
     * layout with qMode.
     */
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers,
            List<VectorRecord> records, int qMode) throws IOException {
        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("Records list cannot be null or empty");
        }
        if (tiers == null || tiers.length == 0 || tiers.length > 8) {
            throw new IllegalArgumentException("Tiers must have between 1 and 8 step boundaries");
        }

        long totalRecords = records.size();

        // 1. Write base .pithos config file containing the 64-byte PLAN header
        Path mainPath = Path.of(basePath);
        try (FileChannel channel = FileChannel.open(mainPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, 64, Arena.global());

            // Magic bytes
            mapped.set(ValueLayout.JAVA_BYTE, 0, (byte) 'P');
            mapped.set(ValueLayout.JAVA_BYTE, 1, (byte) 'L');
            mapped.set(ValueLayout.JAVA_BYTE, 2, (byte) 'A');
            mapped.set(ValueLayout.JAVA_BYTE, 3, (byte) 'N');
            mapped.set(ValueLayout.JAVA_BYTE, 4, planetId);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 5, totalRecords);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 13, planetRadius);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 21, dimension);
            mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 25, tiers.length);
            for (int i = 0; i < tiers.length; i++) {
                mapped.set(ValueLayout.JAVA_INT_UNALIGNED, 29 + (i * 4), tiers[i]);
            }
            // Write qMode to offset 61
            mapped.set(ValueLayout.JAVA_BYTE, 61, (byte) qMode);
            mapped.force();
        }

        // 2. Write IDs file
        Path idsPath = Path.of(basePath + "_ids.bin");
        try (FileChannel channel = FileChannel.open(idsPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * 8, Arena.global());
            for (int i = 0; i < totalRecords; i++) {
                mapped.set(ValueLayout.JAVA_LONG, i * 8, records.get(i).id());
            }
            mapped.force();
        }

        // 3. Write Metadata file (tombstones & attributes, default value is 2 for
        // all-active)
        Path metadataPath = Path.of(basePath + "_metadata.bin");
        try (FileChannel channel = FileChannel.open(metadataPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * 8, Arena.global());
            for (int i = 0; i < totalRecords; i++) {
                // Set metadata to 2 (tombstone bit 0 = 0, attribute mask bit 1 = 1)
                mapped.set(ValueLayout.JAVA_LONG, i * 8, 2L);
            }
            mapped.force();
        }

        // 4. Transform, Binarize, and Write Tier files
        TransformOperator transformer = new TransformOperator(dimension, tiers);
        int numTiers = tiers.length;

        int[] tierLongs = new int[numTiers];
        FileChannel[] tierChannels = new FileChannel[numTiers];
        MemorySegment[] tierMappeds = new MemorySegment[numTiers];

        int prevBound = 0;
        for (int k = 0; k < numTiers; k++) {
            int width = tiers[k] - prevBound;
            tierLongs[k] = width / 64;
            prevBound = tiers[k];

            Path tierPath = Path.of(basePath + "_tier_" + k + ".bin");
            tierChannels[k] = FileChannel.open(tierPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.READ,
                    StandardOpenOption.TRUNCATE_EXISTING);
            long bytesPerRecord = switch (qMode) {
                case 1 -> (width / 4); // 2-bit: 2 bits/dim -> width/4 bytes
                case 2 -> (width * 4L); // Float-Hybrid: raw float32 -> 4 bytes/dim
                default -> (width / 8); // 1-bit: 1 bit/dim -> width/8 bytes
            };
            tierMappeds[k] = tierChannels[k].map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * bytesPerRecord,
                    Arena.global());
        }

        try {
            for (int i = 0; i < totalRecords; i++) {
                VectorRecord rec = records.get(i);
                if (qMode == 1) { // 2-bit mode
                    float[] z = transformer.preconditionAndRotate(rec.vector());
                    float threshold = TransformOperator.calculatePercentileThreshold(z, 0.20f);
                    long[][] packed = transformer.quantize2Bit(z, threshold);
                    long[] signPacked = packed[0];
                    long[] maskPacked = packed[1];

                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        long baseOffset = i * (count * 16L);
                        for (int l = 0; l < count; l++) {
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (l * 8), signPacked[longOffset + l]);
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (count * 8L) + (l * 8),
                                    maskPacked[longOffset + l]);
                        }
                        longOffset += count;
                    }
                } else if (qMode == 2) { // Float-Hybrid: write raw float32 values
                    float[] z = transformer.preconditionAndRotate(rec.vector());
                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k]; // here: dims in this tier
                        int startDim = (k == 0) ? 0 : tiers[k - 1];
                        int width = tiers[k] - startDim;
                        long baseOffset = (long) i * width * 4;
                        for (int l = 0; l < width; l++) {
                            int raw = Float.floatToRawIntBits(z[startDim + l]);
                            tierMappeds[k].set(ValueLayout.JAVA_INT_UNALIGNED, baseOffset + (l * 4), raw);
                        }
                        longOffset += count;
                    }
                } else { // 1-bit mode
                    long[] packed = transformer.transformAndQuantize(rec.vector());
                    int longOffset = 0;
                    for (int k = 0; k < numTiers; k++) {
                        int count = tierLongs[k];
                        long baseOffset = i * (count * 8L);
                        for (int l = 0; l < count; l++) {
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (l * 8), packed[longOffset + l]);
                        }
                        longOffset += count;
                    }
                }
            }
            for (int k = 0; k < numTiers; k++) {
                tierMappeds[k].force();
            }
        } finally {
            for (int k = 0; k < numTiers; k++) {
                if (tierChannels[k] != null) {
                    tierChannels[k].close();
                }
            }
        }

        // 5. Write FP16 sidecar: stores original (pre-rotation) vectors in IEEE 754
        // half-precision.
        // 2 bytes per dimension, row-major layout. Used for high-recall in-engine Stage
        // 2 reranking.
        Path fp16Path = Path.of(basePath + "_fp16.bin");
        try (FileChannel channel = FileChannel.open(fp16Path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            long fp16Bytes = totalRecords * dimension * 2L;
            MemorySegment fp16Mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, fp16Bytes, Arena.global());
            for (int i = 0; i < totalRecords; i++) {
                float[] vec = records.get(i).vector();
                long rowOffset = (long) i * dimension * 2;
                for (int d = 0; d < dimension; d++) {
                    short fp16 = Float.floatToFloat16(vec[d]);
                    fp16Mapped.set(ValueLayout.JAVA_SHORT_UNALIGNED, rowOffset + d * 2L, fp16);
                }
            }
            fp16Mapped.force();
        }
    }

    // =========================================================================
    // CUDA Acceleration Support
    // =========================================================================

    private boolean cudaEnabled = false;
    private int cudaDeviceId = 0;

    public int cudaInit(int deviceId) {
        this.cudaDeviceId = deviceId;
        this.cudaEnabled = true;
        return CudaDeviceManager.initialize(deviceId);
    }

    public void cudaShutdown() {
        CudaDeviceManager.shutdown();
        this.cudaEnabled = false;
    }

    public boolean cudaIsAvailable() {
        return cudaEnabled && CudaDeviceManager.isAvailable() != 0;
    }

    public List<Index.SearchResult>[] cudaBatchSearch(String indexName, float[][] queries, int k) {
        Index index = getIndex(indexName);
        if (index == null) {
            throw new IllegalArgumentException("Index not found: " + indexName);
        }
        return index.cudaBatchSearch(queries, k);
    }

    public long cudaQueryPlanetaryGrid(String indexName, float[][] queries, int[] families, int[] thresholds, MemorySegment votingMask) {
        Index index = getIndex(indexName);
        if (index == null) {
            throw new IllegalArgumentException("Index not found: " + indexName);
        }
        return index.cudaQueryPlanetaryGrid(queries, families, thresholds, votingMask);
    }
}
