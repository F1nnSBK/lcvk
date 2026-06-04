package org.lcvk.vectordb;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Distance metric calculator.
 * Implements 384-bit binary vector Hamming distance.
 * Employs unrolled scalar Long.bitCount for maximum performance on small vectors (48 bytes).
 */
public enum DistanceMetric {
    HAMMING {
        @Override
        public int calculate(long[] a, long[] b) {
            if (a.length != 6 || b.length != 6) {
                throw new IllegalArgumentException("Vectors must be exactly 384 bits (6 longs)");
            }
            int sum = 0;
            sum += Long.bitCount(a[0] ^ b[0]);
            sum += Long.bitCount(a[1] ^ b[1]);
            sum += Long.bitCount(a[2] ^ b[2]);
            sum += Long.bitCount(a[3] ^ b[3]);
            sum += Long.bitCount(a[4] ^ b[4]);
            sum += Long.bitCount(a[5] ^ b[5]);
            return sum;
        }
    };

    /**
     * Calculates the Hamming distance between two on-heap long arrays.
     */
    public abstract int calculate(long[] a, long[] b);

    /**
     * Zero-copy scalar calculation directly between an on-heap query vector
     * and a vector stored off-heap in a MemorySegment at the specified byte offset.
     *
     * @param query       the on-heap query vector (6 longs)
     * @param segment     the off-heap memory segment (file-mapped)
     * @param byteOffset  the starting byte offset of the vector values in the segment
     * @return the Hamming distance
     */
    public static int calculateSegment(long[] query, MemorySegment segment, long byteOffset) {
        int sum = 0;
        sum += Long.bitCount(query[0] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset));
        sum += Long.bitCount(query[1] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + 8));
        sum += Long.bitCount(query[2] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + 16));
        sum += Long.bitCount(query[3] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + 24));
        sum += Long.bitCount(query[4] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + 32));
        sum += Long.bitCount(query[5] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + 40));
        return sum;
    }
}
