package org.pithos;

import java.util.List;

/**
 * Interface representing a binary vector index.
 */
public interface Index extends AutoCloseable {

    /**
     * Inserts a vector record into the index.
     */
    void insert(VectorRecord record);

    /**
     * Searches for the top K closest vectors to the given query vector.
     * Accepts a raw float query vector.
     */
    List<SearchResult> search(float[] query, int k);

    /**
     * Performs a batch search for multiple queries in a single pass.
     * Accepts raw float queries.
     */
    List<SearchResult>[] batchSearch(float[][] queries, int k);

    /**
     * Performs a parallel scan over the index and records resonant matches into a pre-allocated off-heap byte array.
     * Accepts raw float queries.
     */
    long queryPlanetaryGrid(float[][] queries, int[] families, int[] thresholds, java.lang.foreign.MemorySegment votingMask);

    /**
     * Gets the vector dimension of the index.
     */
    int getDimension();

    /**
     * Gets the number of records in the index.
     */
    long size();

    /**
     * Gets the planet ID associated with this index.
     */
    byte getPlanetId();

    /**
     * Gets the equatorial radius of the planet.
     */
    long getPlanetRadius();

    /**
     * Gets the number of Matryoshka tiers in the index.
     */
    int getTierCount();

    /**
     * Represents a single search result match.
     */
    record SearchResult(long id, int score) {}

    @Override
    default void close() throws Exception {}
}
