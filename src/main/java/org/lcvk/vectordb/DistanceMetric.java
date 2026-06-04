package org.lcvk.vectordb;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Distance metric calculator.
 * Implements 384-bit binary vector Hamming distance.
 * Employs Java 25 Vector API (incubator) for hardware SIMD popcount and XOR acceleration.
 */
public enum DistanceMetric {
    HAMMING {
        @Override
        public int calculate(long[] a, long[] b) {
            if (a.length != 6 || b.length != 6) {
                throw new IllegalArgumentException("Vectors must be exactly 384 bits (6 longs)");
            }

            var acc = LongVector.zero(SPECIES);
            int limit = SPECIES.loopBound(a.length);
            int i = 0;
            for (; i < limit; i += SPECIES.length()) {
                var va = LongVector.fromArray(SPECIES, a, i);
                var vb = LongVector.fromArray(SPECIES, b, i);
                var vxor = va.lanewise(VectorOperators.XOR, vb);
                var vpop = vxor.lanewise(VectorOperators.BIT_COUNT);
                acc = acc.add(vpop);
            }
            int sum = (int) acc.reduceLanes(VectorOperators.ADD);
            for (; i < a.length; i++) {
                sum += Long.bitCount(a[i] ^ b[i]);
            }
            return sum;
        }
    };

    private static final VectorSpecies<Long> SPECIES = LongVector.SPECIES_PREFERRED;

    /**
     * Calculates the Hamming distance between two on-heap long arrays.
     */
    public abstract int calculate(long[] a, long[] b);

    /**
     * Zero-copy SIMD calculation directly between an on-heap query vector
     * and a vector stored off-heap in a MemorySegment at the specified byte offset.
     *
     * @param query       the on-heap query vector (6 longs)
     * @param segment     the off-heap memory segment (file-mapped)
     * @param byteOffset  the starting byte offset of the vector values in the segment
     * @return the Hamming distance
     */
    public static int calculateSegment(long[] query, MemorySegment segment, long byteOffset) {
        var acc = LongVector.zero(SPECIES);
        int limit = SPECIES.loopBound(query.length);
        int i = 0;
        for (; i < limit; i += SPECIES.length()) {
            var va = LongVector.fromArray(SPECIES, query, i);
            var vb = LongVector.fromMemorySegment(SPECIES, segment, byteOffset + ((long) i * 8), ByteOrder.nativeOrder());
            var vxor = va.lanewise(VectorOperators.XOR, vb);
            var vpop = vxor.lanewise(VectorOperators.BIT_COUNT);
            acc = acc.add(vpop);
        }
        int sum = (int) acc.reduceLanes(VectorOperators.ADD);
        for (; i < query.length; i++) {
            long val = segment.get(ValueLayout.JAVA_LONG, byteOffset + ((long) i * 8));
            sum += Long.bitCount(query[i] ^ val);
        }
        return sum;
    }
}
