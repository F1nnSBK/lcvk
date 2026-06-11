package org.pithos;

/**
 * Represents a dimension-agnostic vector record containing an ID,
 * a raw float vector, and metadata.
 */
public record VectorRecord(long id, float[] vector, long metadata) {

    public VectorRecord {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }
    }

    public VectorRecord(long id, float[] vector) {
        this(id, vector, 0L);
    }
}
