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

    public VectorDb() {
        this.indices = new ConcurrentHashMap<>();
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
    }

    /**
     * Compiles raw float records into a multi-tier, cache-aligned database file layout.
     */
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers, List<VectorRecord> records) throws IOException {
        compileIndexFile(basePath, planetId, planetRadius, dimension, tiers, records, 0);
    }

    /**
     * Compiles raw float records into a multi-tier, cache-aligned database file layout with qMode.
     */
    public static void compileIndexFile(String basePath, byte planetId, long planetRadius, int dimension, int[] tiers, List<VectorRecord> records, int qMode) throws IOException {
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

        // 3. Write Metadata file (tombstones & attributes, default value is 2 for all-active)
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
            long bytesPerRecord = (qMode == 1) ? (width / 4) : (width / 8);
            tierMappeds[k] = tierChannels[k].map(FileChannel.MapMode.READ_WRITE, 0, totalRecords * bytesPerRecord, Arena.global());
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
                            tierMappeds[k].set(ValueLayout.JAVA_LONG, baseOffset + (count * 8L) + (l * 8), maskPacked[longOffset + l]);
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
    }
}
