
package org.lcvk.vectordb;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

/**
 * Represents a 384-bit packed binary vector record.
 * Uses Project Panama MemoryLayout to define struct offsets and memory sizes.
 *
 * Each record consists of:
 * - A 64-bit ID (long)
 * - 384 bits of vector data (represented as 6 long values)
 * - Total size: 56 bytes
 */
public record VectorRecord(long id, long[] vector) {

    /**
     * Panama StructLayout representation of a VectorRecord.
     */
    public static final StructLayout LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("id"),
            MemoryLayout.sequenceLayout(6, ValueLayout.JAVA_LONG).withName("vector"));

    /**
     * Size in bytes of a single VectorRecord struct (56 bytes).
     */
    public static final long LAYOUT_SIZE = LAYOUT.byteSize();

    public VectorRecord {
        if (vector == null || vector.length != 6) {
            throw new IllegalArgumentException("Vector must consist of exactly 6 long values (384 bits)");
        }
    }
}
