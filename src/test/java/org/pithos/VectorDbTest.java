package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VectorDbTest {

    @Test
    void testJacobiSvdAndEnergy() {
        int D = 4;
        int D0 = 4;
        // Simple diagonal weight matrix W
        float[] weights = {
            2.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.5f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        float[] phi = TransformOperator.computeCumulativeEnergy(weights, D, D0);
        
        // Sum of squared singular values = 4 + 1 + 0.25 = 5.25
        // Cumulative energies:
        // phi[0] = 4 / 5.25 = 0.7619
        // phi[1] = (4 + 1) / 5.25 = 0.9523
        // phi[2] = (4 + 1 + 0.25) / 5.25 = 1.0
        // phi[3] = 1.0
        
        assertEquals(0.7619f, phi[0], 1e-3f);
        assertEquals(0.9523f, phi[1], 1e-3f);
        assertEquals(1.0f, phi[2], 1e-3f);
        assertEquals(1.0f, phi[3], 1e-3f);
    }

    @Test
    void testTransformOperator() {
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] x = new float[128];
        java.util.Arrays.fill(x, 1.0f);
        long[] packed = transformer.transformAndQuantize(x);
        
        assertNotNull(packed);
        assertEquals(2, packed.length);
    }

    @Test
    void testKroneckerTransformOperator() {
        int D = 12;
        int[] tiers = {12}; // Block size 12 -> non-power of two. Triggers Kronecker fallback.
        TransformOperator transformer = new TransformOperator(D, tiers);

        float[] x = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
        long[] packed = transformer.transformAndQuantize(x);
        
        assertNotNull(packed);
        assertEquals(1, packed.length);
    }

    @Test
    void testCompileAndQueryIndex(@TempDir Path tempDir) throws IOException {
        Path dbPath = tempDir.resolve("test_pithos");
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        // Define target transformed vectors z with positive MSB (z[127] >= 0.0) to bypass QEG
        // MSB in flat index logic corresponds to bit 63 of the first tier (index 63 of transformed vector z)
        float[] targetZ0 = new float[128];
        java.util.Arrays.fill(targetZ0, -1.0f);
        targetZ0[63] = 1.0f; // Enable QEG bit

        float[] targetZ1 = new float[128];
        java.util.Arrays.fill(targetZ1, 1.0f); // All positive -> closest to query

        float[] targetZ2 = new float[128];
        java.util.Arrays.fill(targetZ2, -0.2f);
        targetZ2[63] = 1.0f; // Enable QEG bit

        // Back-project target transform vectors to raw input vectors x
        float[] vec0 = transformer.backProject(targetZ0);
        float[] vec1 = transformer.backProject(targetZ1);
        float[] vec2 = transformer.backProject(targetZ2);

        List<VectorRecord> records = List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1),
            new VectorRecord(2, vec2)
        );

        // Compile index
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, D, tiers, records);

        // Load index without weights (equal energy distribution)
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("pithos_test", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(3, index.size());
        assertEquals(128, index.getDimension());
        assertEquals(1, index.getPlanetId());
        assertEquals(1737400L, index.getPlanetRadius());
        assertEquals(2, index.getTierCount());

        // Perform float-based KNN search using a query that is all-positive in transformed domain
        float[] targetQueryZ = new float[128];
        java.util.Arrays.fill(targetQueryZ, 0.9f);
        float[] query = transformer.backProject(targetQueryZ);

        List<Index.SearchResult> results = index.search(query, 2);
        
        assertEquals(2, results.size());
        // Closest should be vec1 (ID 1) because targetZ1 has all positive entries (aligned with query)
        assertEquals(1, results.get(0).id());

        // Clean up
        db.close();
    }

    @Test
    void testCompileAndQuery2BitIndex(@TempDir Path tempDir) throws IOException {
        Path dbPath = tempDir.resolve("test_pithos_2bit");
        int D = 128;
        int[] tiers = {64, 128};
        TransformOperator transformer = new TransformOperator(D, tiers);

        // We want to verify that 2-bit mode successfully indexes and filters query noise
        float[] vec0 = new float[128];
        java.util.Arrays.fill(vec0, 0.5f);
        // Ensure MSB is positive for QEG
        float[] targetZ0 = transformer.preconditionAndRotate(vec0);
        targetZ0[63] = 1.0f;
        vec0 = transformer.backProject(targetZ0);

        float[] vec1 = new float[128];
        java.util.Arrays.fill(vec1, -0.5f);
        float[] targetZ1 = transformer.preconditionAndRotate(vec1);
        targetZ1[63] = 1.0f;
        vec1 = transformer.backProject(targetZ1);

        List<VectorRecord> records = List.of(
            new VectorRecord(0, vec0),
            new VectorRecord(1, vec1)
        );

        // Compile index with qMode = 1 (2-bit)
        VectorDb.compileIndexFile(dbPath.toString(), (byte) 1, 1737400L, D, tiers, records, 1);

        // Load index
        VectorDb db = new VectorDb();
        Index index = db.loadIndex("pithos_2bit_test", dbPath.toString(), null, 0);

        assertNotNull(index);
        assertEquals(2, index.size());

        // Search with query vec0
        List<Index.SearchResult> results = index.search(vec0, 2);
        assertEquals(2, results.size());
        assertEquals(0, results.get(0).id()); // Closest should be ID 0

        db.close();
    }
}
