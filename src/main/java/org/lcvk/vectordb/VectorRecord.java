
package org.lcvk.vectordb;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

/**
 * Represents a 64-byte Cache-Line aligned 384-bit packed binary vector record.
 */
public record VectorRecord(long id, long[] vector, long metadata) {

    /**
     * Panama StructLayout representation of a VectorRecord.
     */
    public static final StructLayout LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_LONG.withName("id"),
            MemoryLayout.sequenceLayout(6, ValueLayout.JAVA_LONG).withName("vector"),
            ValueLayout.JAVA_LONG.withName("metadata"));

    /**
     * Size in bytes of a single VectorRecord struct (64 bytes).
     */
    public static final long LAYOUT_SIZE = 64;

    public VectorRecord {
        if (vector == null || vector.length != 6) {
            throw new IllegalArgumentException("Vector must consist of exactly 6 long values (384 bits)");
        }
    }

    public VectorRecord(long id, long[] vector) {
        this(id, vector, 0L);
    }
}
