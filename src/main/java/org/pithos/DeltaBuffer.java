package org.pithos;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Log-Structured Merge (LSM) Delta-Buffer for real-time inserts into the Pithos engine.
 *
 * <p>Newly inserted vectors land in this in-memory flat buffer. Searches query both the
 * base index and this buffer in parallel, merging results before returning. When the buffer
 * size reaches {@link #flushThreshold}, it can be serialized and merged into the base index.
 *
 * <p>Thread safety: concurrent reads and writes are supported via a ReentrantReadWriteLock.
 */
public class DeltaBuffer {

    /** A single buffered entry: record ID, original float vector, tombstone flag. */
    private record BufferEntry(long id, float[] vector, boolean tombstone) {}

    private final int dimension;
    private final int flushThreshold;

    /** Ordered list of inserted entries (append-only, tombstones included). */
    private final List<BufferEntry> entries;

    /** Count of live (non-tombstoned) entries. */
    private final AtomicInteger liveCount = new AtomicInteger(0);

    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /**
     * @param dimension      vector dimension (must match the base index)
     * @param flushThreshold soft threshold for live-entry count above which flush is recommended
     */
    public DeltaBuffer(int dimension, int flushThreshold) {
        this.dimension = dimension;
        this.flushThreshold = flushThreshold;
        this.entries = new CopyOnWriteArrayList<>();
    }

    /**
     * Inserts a new vector record into the delta buffer.
     */
    public void insert(long id, float[] vector) {
        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                    "Vector dimension mismatch: expected " + dimension + ", got " + vector.length);
        }
        lock.writeLock().lock();
        try {
            entries.add(new BufferEntry(id, vector.clone(), false));
            liveCount.incrementAndGet();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Soft-deletes a record (tombstone). All entries with the given ID are marked deleted.
     *
     * @return true if at least one live entry was tombstoned
     */
    public boolean delete(long id) {
        lock.writeLock().lock();
        try {
            boolean found = false;
            for (int i = 0; i < entries.size(); i++) {
                BufferEntry e = entries.get(i);
                if (e.id() == id && !e.tombstone()) {
                    entries.set(i, new BufferEntry(e.id(), e.vector(), true));
                    liveCount.decrementAndGet();
                    found = true;
                }
            }
            return found;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Number of live (non-tombstoned) entries. */
    public int liveSize() { return liveCount.get(); }

    /** Total entries including tombstones. */
    public int totalSize() { return entries.size(); }

    /** Returns true if live count has reached or exceeded the flush threshold. */
    public boolean needsFlush() { return liveCount.get() >= flushThreshold; }

    /**
     * Searches the delta buffer for the top-K nearest neighbors to the given query.
     * Uses exact L2 distance (no quantization) on the original float vectors.
     */
    public List<Index.SearchResult> searchKnn(float[] query, int k) {
        if (k <= 0 || liveCount.get() == 0) {
            return List.of();
        }
        // Max-heap of size k keyed by distance bits (for efficient eviction of worst candidate)
        PriorityQueue<long[]> heap = new PriorityQueue<>(
                (a, b) -> Long.compare(b[0], a[0]));

        lock.readLock().lock();
        try {
            for (BufferEntry e : entries) {
                if (e.tombstone()) continue;
                float dist = exactL2(query, e.vector());
                long distBits = Float.floatToRawIntBits(dist) & 0xFFFFFFFFL;
                if (heap.size() < k) {
                    heap.offer(new long[]{distBits, e.id()});
                } else if (distBits < heap.peek()[0]) {
                    heap.poll();
                    heap.offer(new long[]{distBits, e.id()});
                }
            }
        } finally {
            lock.readLock().unlock();
        }

        List<Index.SearchResult> results = new ArrayList<>(heap.size());
        while (!heap.isEmpty()) {
            long[] entry = heap.poll();
            float d = Float.intBitsToFloat((int) entry[0]);
            results.add(new Index.SearchResult(entry[1], (int) (d * 1_000_000f)));
        }
        results.sort((a, b) -> Integer.compare(a.score(), b.score()));
        return results;
    }

    private static float exactL2(float[] a, float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Serializes all live entries to a binary file for backup or offline merge.
     *
     * <p>File format (big-endian):
     * <pre>
     *   [int]  dimension
     *   [int]  num_live_entries
     *   for each entry:
     *     [long]   id
     *     [float]  vector[0..dimension-1]
     * </pre>
     */
    public void serializeToPath(String path) throws IOException {
        lock.readLock().lock();
        try (DataOutputStream out = new DataOutputStream(
                Files.newOutputStream(Path.of(path)))) {
            List<BufferEntry> snapshot = new ArrayList<>();
            for (BufferEntry e : entries) {
                if (!e.tombstone()) snapshot.add(e);
            }
            out.writeInt(dimension);
            out.writeInt(snapshot.size());
            for (BufferEntry e : snapshot) {
                out.writeLong(e.id());
                for (float v : e.vector()) {
                    out.writeFloat(v);
                }
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Deserializes a DeltaBuffer from a previously serialized binary file.
     */
    public static DeltaBuffer deserializeFromPath(String path, int flushThreshold) throws IOException {
        try (DataInputStream in = new DataInputStream(
                Files.newInputStream(Path.of(path)))) {
            int dim = in.readInt();
            int numEntries = in.readInt();
            DeltaBuffer buf = new DeltaBuffer(dim, flushThreshold);
            for (int i = 0; i < numEntries; i++) {
                long id = in.readLong();
                float[] vec = new float[dim];
                for (int d = 0; d < dim; d++) {
                    vec[d] = in.readFloat();
                }
                buf.insert(id, vec);
            }
            return buf;
        }
    }

    /**
     * Drains and returns all live entries, clearing the buffer.
     * Should be called when merging the delta into the base index.
     */
    public List<VectorRecord> drainLiveEntries() {
        lock.writeLock().lock();
        try {
            List<VectorRecord> result = new ArrayList<>();
            for (BufferEntry e : entries) {
                if (!e.tombstone()) {
                    result.add(new VectorRecord(e.id(), e.vector()));
                }
            }
            entries.clear();
            liveCount.set(0);
            return result;
        } finally {
            lock.writeLock().unlock();
        }
    }
}
