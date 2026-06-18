package examples.java;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A demo showcasing how Pithos achieves Zero-Cost Abstractions using the
 * Java Foreign Function & Memory (FFM) API (Project Panama) available in Java 25.
 * 
 * This demo shows:
 * 1. Allocation of off-heap memory outside the JVM Garbage Collector.
 * 2. Deterministic memory lifecycle management using Arena.
 * 3. Structured, highly optimized memory access using ValueLayouts.
 */
public class ZeroCostDemo {

    public static void main(String[] args) {
        System.out.println("=== Pithos Zero-Cost Abstraction Demo (Java FFM API) ===");

        // 1. Allocate off-heap memory inside a confined arena
        // Confined arenas are bound to a single thread and can be closed deterministically.
        // When the try-with-resources block exits, the arena is closed and the memory is freed.
        try (Arena arena = Arena.ofConfined()) {
            long numRecords = 1000;
            int wordCount = 6; // 6 * 64 bits = 384 dimensions
            long sizeBytes = numRecords * wordCount * 8;

            System.out.printf("Allocating %d bytes off-heap for %d records...\n", sizeBytes, numRecords);
            MemorySegment segment = arena.allocate(sizeBytes);

            // 2. Initialize off-heap memory with synthetic binary data
            // We use ValueLayout.JAVA_LONG for fast, unmanaged writes
            System.out.println("Populating off-heap columnar tier files...");
            for (long i = 0; i < numRecords; i++) {
                for (int w = 0; w < wordCount; w++) {
                    long byteOffset = (i * wordCount + w) * 8;
                    // Write a synthetic deterministic pattern (e.g. alternating bits)
                    segment.set(ValueLayout.JAVA_LONG, byteOffset, i ^ w);
                }
            }

            // 3. Perform a zero-copy simulated query sweep
            long[] query = new long[]{0xF0F0F0F0F0F0F0F0L, 0x0F0F0F0F0F0F0F0FL, 0xAAAAAAAAAAAAAAAAL, 0x5555555555555555L, 0x0L, 0xFFFFFFFFFFFFFFFFL};
            System.out.println("Executing query sweep over off-heap memory segment...");
            
            long startTime = System.nanoTime();
            long matchCount = 0;
            
            for (long i = 0; i < numRecords; i++) {
                int hammingDistance = 0;
                for (int w = 0; w < wordCount; w++) {
                    long byteOffset = (i * wordCount + w) * 8;
                    long recordWord = segment.get(ValueLayout.JAVA_LONG, byteOffset);
                    hammingDistance += Long.bitCount(query[w] ^ recordWord);
                }
                // Check if Hamming distance meets target threshold (e.g., <= 192 bits out of 384)
                if (hammingDistance <= 192) {
                    matchCount++;
                }
            }
            
            long endTime = System.nanoTime();
            double elapsedMs = (endTime - startTime) / 1_000_000.0;
            
            System.out.printf("Scan completed in %.3f ms. Found %d matching records.\n", elapsedMs, matchCount);
            System.out.println("Closing the Arena... Memory will be freed immediately without GC involvement.");
        } // The Arena is automatically closed here and native memory is reclaimed.

        System.out.println("Arena closed successfully. Demo finished!");
    }
}
