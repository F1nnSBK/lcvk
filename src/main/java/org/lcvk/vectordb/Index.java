package org.lcvk.vectordb;

import java.util.List;

/**
 * Interface representing a binary vector index.
 */
public interface Index extends AutoCloseable {
    /**
     * Inserts a vector record into the index.
     * Note: For memory-mapped files, this might write to the file or throw if read-only.
     *
     * @param record the vector record to insert
     */
    void insert(VectorRecord record);

    /**
     * Searches for the top K closest vectors to the given query vector.
     *
     * @param query the query vector (6 longs)
     * @param k     the number of neighbors to return
     * @return list of search results sorted by distance (lowest score first)
     */
    List<SearchResult> search(long[] query, int k);

    /**
     * Performs a batch search for multiple queries in a single pass.
     * Implements multi-query scanning to maximize L1/L2 cache locality.
     *
     * @param queries array of query vectors
     * @param k       number of nearest neighbors to return per query
     * @return array of search result lists corresponding to each query
     */
    List<SearchResult>[] batchSearch(long[][] queries, int k);

    /**
     * Performs a parallel scan over the index and records resonant matches into a pre-allocated off-heap byte array.
     *
     * @param queries    array of 384-bit query vectors (each 6 longs)
     * @param families   array mapping each query to a family index (0-7)
     * @param thresholds array mapping each query to a maximum distance threshold
     * @param votingMask pre-allocated off-heap MemorySegment of size equal to the index size
     * @return the number of resonant tiles (tiles with >= 7 set bits, i.e., popcount >= 7)
     */
    long queryPlanetaryGrid(long[][] queries, int[] families, int[] thresholds, java.lang.foreign.MemorySegment votingMask);

    /**
     * Gets the expected dimension of vectors in this index (384).
     *
     * @return the vector dimension
     */
    int getDimension();

    /**
     * Gets the number of vectors stored in the index.
     *
     * @return number of vectors
     */
    long size();

    /**
     * Represents a single search result match.
     *
     * @param id    the vector ID
     * @param score the Hamming distance (lower is closer)
     */
    record SearchResult(long id, int score) {}

    @Override
    default void close() throws Exception {}
}
