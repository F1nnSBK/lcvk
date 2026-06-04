package org.lcvk.vectordb;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VectorDbTest {

    @Test
    void testHammingDistanceMetrics() {
        // 6 longs representing 384 bits
        long[] v1 = {0b1111L, 0L, 0L, 0L, 0L, 0L};
        long[] v2 = {0b1010L, 0L, 0L, 0L, 0L, 0L}; // differences: bits at index 0 and 2 are same (0), index 1 and 3 are different (1). Total differences: 2 bits.

        assertEquals(2, DistanceMetric.HAMMING.calculate(v1, v2));

        // Let's test with MemorySegment
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(8 * 6);
            for (int i = 0; i < 6; i++) {
                segment.set(ValueLayout.JAVA_LONG, i * 8, v2[i]);
            }

            int dist = DistanceMetric.calculateSegment(v1, segment, 0);
            assertEquals(2, dist);
        }
    }

    @Test
    void testCompileAndQueryIndex(@TempDir Path tempDir) throws IOException {
        Path dbPath = tempDir.resolve("test_binary.vdb");

        // Create mock records
        // Record 100: all zeros
        long[] vec0 = new long[]{0L, 0L, 0L, 0L, 0L, 0L};
        // Record 101: 4 bits set in first long
        long[] vec1 = new long[]{0b1111L, 0L, 0L, 0L, 0L, 0L};
        // Record 102: 2 bits set in first long
        long[] vec2 = new long[]{0b0011L, 0L, 0L, 0L, 0L, 0L};

        List<VectorRecord> records = List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1),
            new VectorRecord(2, vec2)
        );

        // Compile index to file
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, records);

        // Load index via memory-mapping
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("lunar_test", dbPath.toString());

        assertNotNull(index);
        assertEquals(3, index.size());
        assertEquals(384, index.getDimension());

        // Query: close to vec2 (0b0011L)
        // Let's query with 0b0010L (1 bit diff from vec2, 1 bit diff from vec0, 3 bits diff from vec1)
        long[] query = new long[]{0b0010L, 0L, 0L, 0L, 0L, 0L};

        List<Index.SearchResult> results = index.search(query, 3);
        assertEquals(3, results.size());

        // Expected sorted order:
        // vec0 (dist = 1)
        // vec2 (dist = 1)
        // vec1 (dist = 3)
        // (Due to tie-breaker, ID 100 will come before 102 or vice versa based on sorting logic)
        assertEquals(1, results.get(0).score());
        assertEquals(1, results.get(1).score());
        assertEquals(3, results.get(2).score());

        // Test batch search with multiple queries
        long[][] batchQueries = new long[][]{
            new long[]{0b1111L, 0L, 0L, 0L, 0L, 0L}, // matches vec1 perfectly (dist = 0)
            new long[]{0L, 0L, 0L, 0L, 0L, 0L}       // matches vec0 perfectly (dist = 0)
        };

        List<Index.SearchResult>[] batchResults = index.batchSearch(batchQueries, 1);
        assertEquals(2, batchResults.length);
        assertEquals(1, batchResults[0].get(0).id());
        assertEquals(0, batchResults[0].get(0).score());

        assertEquals(0, batchResults[1].get(0).id());
        assertEquals(0, batchResults[1].get(0).score());
    }
}
