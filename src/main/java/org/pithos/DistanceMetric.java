package org.pithos;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Dimension-agnostic Hamming distance metric calculator.
 */
public enum DistanceMetric {
    HAMMING {
        @Override
        public int calculate(long[] a, long[] b) {
            int sum = 0;
            int len = Math.min(a.length, b.length);
            for (int i = 0; i < len; i++) {
                sum += Long.bitCount(a[i] ^ b[i]);
            }
            return sum;
        }
    };

    public abstract int calculate(long[] a, long[] b);

    /**
     * Calculates the Hamming distance between an on-heap query vector and a memory segment offset.
     */
    public static int calculateSegment(long[] query, MemorySegment segment, long byteOffset, int numLongs) {
        int sum = 0;
        for (int i = 0; i < numLongs; i++) {
            sum += Long.bitCount(query[i] ^ segment.get(ValueLayout.JAVA_LONG, byteOffset + (i * 8)));
        }
        return sum;
    }
}
