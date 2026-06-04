package org.lcvk.vectordb;

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
 * Main coordinator for the vector database.
 * Manages active Memory-Mapped Index instances and provides tools to compile indexes to files.
 */
public class VectorDb {
    private final Map<String, Index> indices;

    public VectorDb() {
        this.indices = new ConcurrentHashMap<>();
    }

    /**
     * Maps an existing database file off-heap and registers it.
     *
     * @param name index register name
     * @param path file path to the database file
     * @return the mapped Index
     * @throws IOException if mapping fails
     */
    public Index loadIndex(String name, String path) throws IOException {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Index name cannot be empty");
        }
        Index index = FlatIndex.mapFile(path);
        indices.put(name, index);
        return index;
    }

    /**
     * Retrieves a registered index by name.
     */
    public Index getIndex(String name) {
        return indices.get(name);
    }

    /**
     * Unregisters an index.
     */
    public boolean dropIndex(String name) {
        return indices.remove(name) != null;
    }

    /**
     * Offline Index compiler.
     * Writes out a binary file containing a 64-byte PLAN header followed by 48-byte vectors.
     * The vectors are placed at offsets corresponding to their Geographic IDs.
     *
     * @param path         path to write the database file to
     * @param planetId     ID of the planet (0=Earth, 1=Moon, 2=Mars, 3=Venus)
     * @param planetRadius equatorial radius in meters
     * @param records      list of vector records containing geographic IDs and vectors
     * @throws IOException if writing fails
     */
    public static void compileIndexFile(String path, byte planetId, long planetRadius, List<VectorRecord> records) throws IOException {
        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("Records list cannot be null or empty");
        }
        Path filePath = Path.of(path);

        // Find the maximum geographic ID to determine total tile count
        long maxId = 0;
        for (VectorRecord record : records) {
            if (record.id() > maxId) {
                maxId = record.id();
            }
        }
        long totalTiles = maxId + 1;
        long totalBytes = 64 + totalTiles * 48L; // 64-byte header + totalTiles * 48 bytes

        try (FileChannel channel = FileChannel.open(filePath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.READ,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            // Map standard read-write FFM MemorySegment
            MemorySegment mapped = channel.map(FileChannel.MapMode.READ_WRITE, 0, totalBytes, Arena.global());

            // Write 64-byte Header
            mapped.set(ValueLayout.JAVA_BYTE, 0, (byte) 'P');
            mapped.set(ValueLayout.JAVA_BYTE, 1, (byte) 'L');
            mapped.set(ValueLayout.JAVA_BYTE, 2, (byte) 'A');
            mapped.set(ValueLayout.JAVA_BYTE, 3, (byte) 'N');
            mapped.set(ValueLayout.JAVA_BYTE, 4, planetId);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 5, totalTiles);
            mapped.set(ValueLayout.JAVA_LONG_UNALIGNED, 13, planetRadius);
            // Bytes 21-63 are zero-filled automatically

            // Write vector records into offsets dictated by their ID (Geographic ID mapping)
            for (VectorRecord record : records) {
                long offset = 64 + record.id() * 48L;
                long[] vector = record.vector();
                for (int j = 0; j < 6; j++) {
                    mapped.set(ValueLayout.JAVA_LONG, offset + (j * 8), vector[j]);
                }
            }
            mapped.force(); // Commit changes to physical disk
        }
    }
}
