package org.pithos;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CLongPointer;
import org.graalvm.nativeimage.c.type.CFloatPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * C-API entry points for the Pithos binary vector database.
 * Exposes methods to native callers using GraalVM {@link CEntryPoint}.
 *
 * <p>
 * <h3>Memory Management Guidelines:</h3>
 * <ul>
 * <li>The C caller is responsible for allocating and freeing all pointers
 * passed for outputs
 * (e.g., {@code outIds}, {@code outDistances}, {@code votingMask}).</li>
 * <li>Pithos loads indices off-heap via memory-mapping ({@code mmap}), keeping
 * them in read-only
 * virtual memory segments. These are freed when the database is closed or the
 * index is dropped.</li>
 * <li>Direct off-heap memory addresses can be retrieved via
 * {@link #getTierAddress} for direct
 * FPGA DMA transfers or GPU execution.</li>
 * </ul>
 *
 * <p>
 * <h3>API Return Codes:</h3>
 * <table>
 * <tr>
 * <th>Code</th>
 * <th>Name</th>
 * <th>Description</th>
 * </tr>
 * <tr>
 * <td>0</td>
 * <td>SUCCESS</td>
 * <td>Operation completed successfully.</td>
 * </tr>
 * <tr>
 * <td>-1</td>
 * <td>ERR_DB_NOT_INIT</td>
 * <td>The global database coordinator is not initialized. Call {@link #init}
 * first.</td>
 * </tr>
 * <tr>
 * <td>-2</td>
 * <td>ERR_INDEX_NOT_FOUND</td>
 * <td>The requested index name is not mapped.</td>
 * </tr>
 * <tr>
 * <td>-3</td>
 * <td>ERR_INVALID_OPERATION</td>
 * <td>The operation is not supported by this index type (e.g., chunking on a
 * non-Flat index).</td>
 * </tr>
 * <tr>
 * <td>-4</td>
 * <td>ERR_INTERNAL_EXCEPTION</td>
 * <td>An unexpected internal Java exception occurred. Trace printed to
 * stderr.</td>
 * </tr>
 * <tr>
 * <td>-5</td>
 * <td>ERR_FILE_IO</td>
 * <td>Could not read or write file(s) on the disk.</td>
 * </tr>
 * <tr>
 * <td>-6</td>
 * <td>ERR_UNSUPPORTED_LAYOUT</td>
 * <td>The index structure/layout does not match the operation.</td>
 * </tr>
 * </table>
 */
public class CApi {
    private static VectorDb db;

    private CApi() {
    }

    /**
     * Initializes the global database coordinator.
     * Must be called once before performing any database operations.
     *
     * @param thread the GraalVM isolate thread context
     * @return 0 on success, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_init")
    public static int init(IsolateThread thread) {
        try {
            db = new VectorDb();
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Maps an existing compiled database off-heap into virtual memory without
     * custom weights.
     * Use this when the index was compiled with a static energy layout, or when
     * using an
     * equal energy distribution fallback.
     *
     * @param thread the GraalVM isolate thread context
     * @param name   C string specifying the unique logical name to register the
     *               loaded index under
     * @param path   C string specifying the base filepath of the compiled index on
     *               disk
     * @return 0 on success, -1 if database not initialized, -5 on File I/O error,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_load_index")
    public static int loadIndex(IsolateThread thread, CCharPointer name, CCharPointer path) {
        if (db == null) {
            return -1;
        }
        try {
            String indexName = CTypeConversion.toJavaString(name);
            String filePath = CTypeConversion.toJavaString(path);
            db.loadIndex(indexName, filePath, null, 0);
            return 0;
        } catch (IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Maps an existing compiled database off-heap, supplying frozen model weights.
     * The model weights are used to compute SVD singular values, constructing the
     * Matryoshka
     * energy cumulative distribution to dynamically target a recall energy budget
     * during queries.
     *
     * @param thread  the GraalVM isolate thread context
     * @param name    C string specifying the unique logical name to register the
     *                loaded index under
     * @param path    C string specifying the base filepath of the compiled index on
     *                disk
     * @param weights C float array pointer storing the frozen projection/LoRA
     *                weights matrix of size {@code (dimension * loraDim)}
     * @param loraDim the inner bottleneck dimension of the LoRA weights matrix (D0)
     * @return 0 on success, -1 if database not initialized, -5 on File I/O error,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_load_index_with_weights")
    public static int loadIndexWithWeights(IsolateThread thread, CCharPointer name, CCharPointer path,
            CFloatPointer weights, int loraDim) {
        if (db == null) {
            return -1;
        }
        try {
            String indexName = CTypeConversion.toJavaString(name);
            String filePath = CTypeConversion.toJavaString(path);

            Index tempIdx = FlatIndex.mapFile(filePath, null, 0);
            int dim = tempIdx.getDimension();
            tempIdx.close();

            float[] javaWeights = new float[dim * loraDim];
            for (int i = 0; i < javaWeights.length; i++) {
                javaWeights[i] = weights.read(i);
            }

            db.loadIndex(indexName, filePath, javaWeights, loraDim);
            return 0;
        } catch (IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Retrieves metadata attributes for a loaded index.
     *
     * @param thread          the GraalVM isolate thread context
     * @param indexName       C string identifying the target loaded index
     * @param outDimension    output pointer populated with the vector
     *                        dimensionality (D)
     * @param outSize         output pointer populated with the total record count
     *                        (N)
     * @param outPlanetId     output pointer populated with the associated Planet ID
     *                        code
     * @param outPlanetRadius output pointer populated with the associated Planet
     *                        Radius in meters
     * @param outTiersCount   output pointer populated with the total number of
     *                        cumulative search tiers
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_get_info")
    public static int getInfo(IsolateThread thread, CCharPointer indexName,
            CIntPointer outDimension, CLongPointer outSize,
            CCharPointer outPlanetId, CLongPointer outPlanetRadius,
            CIntPointer outTiersCount) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }

            outDimension.write(0, index.getDimension());
            outSize.write(0, index.size());
            outPlanetId.write(0, (byte) index.getPlanetId());
            outPlanetRadius.write(0, index.getPlanetRadius());
            outTiersCount.write(0, index.getTierCount());
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Performs a batch KNN search on raw float query vectors.
     * Evaluates candidates off-heap and returns their resolved record IDs and
     * scores.
     *
     * @param thread       the GraalVM isolate thread context
     * @param indexName    C string identifying the target index to scan
     * @param queries      contiguous C float array pointer storing query vectors of
     *                     size {@code (numQueries * dimension)}
     * @param numQueries   number of query vectors in the batch
     * @param k            number of nearest neighbors to retrieve per query (top-K)
     * @param outIds       pre-allocated C long array pointer of size
     *                     {@code (numQueries * k)} to store output record IDs
     * @param outDistances pre-allocated C int array pointer of size
     *                     {@code (numQueries * k)} to store output metric distances
     *                     (scaled by 1,000,000)
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_batch_search")
    public static int batchSearch(IsolateThread thread, CCharPointer indexName, CFloatPointer queries, int numQueries,
            int k,
            CLongPointer outIds, CIntPointer outDistances) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }

            int dim = index.getDimension();
            float[][] javaQueries = new float[numQueries][dim];
            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
            }

            List<Index.SearchResult>[] results = index.batchSearch(javaQueries, k);

            for (int q = 0; q < numQueries; q++) {
                List<Index.SearchResult> queryResults = results[q];
                int count = queryResults.size();
                for (int i = 0; i < k; i++) {
                    long outId = -1;
                    int outDist = Integer.MAX_VALUE;
                    if (i < count) {
                        Index.SearchResult r = queryResults.get(i);
                        outId = r.id();
                        outDist = r.score();
                    }
                    outIds.write(q * k + i, outId);
                    outDistances.write(q * k + i, outDist);
                }
            }
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Performs a batch search and outputs matching candidate votes into a
     * pre-allocated voting mask.
     * Designed for multi-family resonant voting across scientific criteria.
     *
     * @param thread          the GraalVM isolate thread context
     * @param indexName       C string identifying the target index
     * @param queries         contiguous C float array pointer storing query vectors
     *                        of size {@code (numQueries * dimension)}
     * @param queryFamilies   C int array pointer of size {@code (numQueries)}
     *                        defining the semantic family index (0-7) for each
     *                        query
     * @param queryThresholds C int array pointer of size {@code (numQueries)}
     *                        defining the Hamming distance threshold for each query
     * @param numQueries      number of queries in the batch
     * @param votingMask      pre-allocated C byte array pointer of size
     *                        {@code (totalRecords)} that accumulates family
     *                        bitmasks (OR-ed)
     * @return the number of resonant records (i.e., records with a non-zero voting
     *         mask), or negative error code on failure
     */
    @CEntryPoint(name = "vdb_query_planetary_grid")
    public static long queryPlanetaryGrid(IsolateThread thread, CCharPointer indexName, CFloatPointer queries,
            CIntPointer queryFamilies, CIntPointer queryThresholds, int numQueries,
            CCharPointer votingMask) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }

            int dim = index.getDimension();
            long totalTiles = index.size();

            float[][] javaQueries = new float[numQueries][dim];
            int[] javaFamilies = new int[numQueries];
            int[] javaThresholds = new int[numQueries];

            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
                javaFamilies[q] = queryFamilies.read(q);
                javaThresholds[q] = queryThresholds.read(q);
            }

            long rawAddress = votingMask.rawValue();
            MemorySegment maskSegment = MemorySegment.ofAddress(rawAddress).reinterpret(totalTiles);

            return index.queryPlanetaryGrid(javaQueries, javaFamilies, javaThresholds, maskSegment);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Compiles raw float records into a multi-tier database file layout with
     * configurable quantization.
     * This method writes the core config header, IDs file, metadata file, tier
     * binary tables, and
     * the FP16 sidecar file for Stage 2 reranking.
     *
     * @param thread       the GraalVM isolate thread context
     * @param path         C string base filepath to compile the index files to
     * @param planetId     associated planetary target identifier code (e.g., 1 for
     *                     Moon, 2 for Mars)
     * @param planetRadius radius of the target planet in meters (used for spatial
     *                     projection checks)
     * @param dimension    dimensionality of the input float vectors
     * @param tiers        C int array defining dimension boundaries for each
     *                     Matryoshka cumulative energy tier
     * @param numTiers     number of cumulative energy tiers defined
     * @param ids          C long array containing unique identifiers for each
     *                     record
     * @param vectors      C float array storing the raw vector representations of
     *                     size {@code (numRecords * dimension)}
     * @param numRecords   total count of vectors to compile
     * @param qMode        quantization mode: 0 = 1-bit sign, 1 = 2-bit ternary, 2 =
     *                     float32 bypass
     * @return 0 on success, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_compile_index_file")
    public static int compileIndexFile(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
            int dimension, CIntPointer tiers, int numTiers,
            CLongPointer ids, CFloatPointer vectors, int numRecords, int qMode) {
        try {
            String filePath = CTypeConversion.toJavaString(path);

            int[] javaTiers = new int[numTiers];
            for (int i = 0; i < numTiers; i++) {
                javaTiers[i] = tiers.read(i);
            }

            List<VectorRecord> records = new ArrayList<>(numRecords);
            for (int i = 0; i < numRecords; i++) {
                long id = ids.read(i);
                float[] vector = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    vector[j] = vectors.read(i * dimension + j);
                }
                records.add(new VectorRecord(id, vector));
            }

            VectorDb.compileIndexFile(filePath, planetId, planetRadius, dimension, javaTiers, records, qMode);
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Compiles raw float records into a multi-tier database file layout with optional FP16 sidecar.
     *
     * @param thread            the GraalVM isolate thread context
     * @param path              destination base path for compiled files
     * @param planetId          planet ID code
     * @param planetRadius      planet radius in meters
     * @param dimension         vector dimensionality (D)
     * @param tiers             cumulative search tier step boundaries array
     * @param numTiers          total number of cumulative tiers (must be <= 8)
     * @param ids               record identifier array
     * @param vectors           contiguous query vector array
     * @param numRecords        total number of vectors
     * @param qMode             quantization mode (0=1-bit, 1=2-bit, 2=FP32)
     * @param writeFp16         whether to write FP16 sidecar (1=true, 0=false)
     * @return 0 on success, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_compile_index_file_ext")
    public static int compileIndexFileExt(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
            int dimension, CIntPointer tiers, int numTiers,
            CLongPointer ids, CFloatPointer vectors, int numRecords, int qMode, int writeFp16) {
        try {
            String filePath = CTypeConversion.toJavaString(path);

            int[] javaTiers = new int[numTiers];
            for (int i = 0; i < numTiers; i++) {
                javaTiers[i] = tiers.read(i);
            }

            List<VectorRecord> records = new ArrayList<>(numRecords);
            for (int i = 0; i < numRecords; i++) {
                long id = ids.read(i);
                float[] vector = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    vector[j] = vectors.read(i * dimension + j);
                }
                records.add(new VectorRecord(id, vector));
            }

            VectorDb.compileIndexFile(filePath, planetId, planetRadius, dimension, javaTiers, records, qMode, writeFp16 != 0);
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Returns the total record count (N) for a loaded index.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the index
     * @return total record count on success, -1 if database not initialized, or -2
     *         if index not found
     */
    @CEntryPoint(name = "vdb_size")
    public static long size(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        Index index = db.getIndex(idxName);
        if (index == null) {
            return -2;
        }
        return index.size();
    }

    /**
     * Unmaps and drops an index from the database memory space, releasing all
     * associated file mappings.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the index to drop
     * @return 0 on success, -1 if database not initialized, or -2 if index not
     *         found
     */
    @CEntryPoint(name = "vdb_drop_index")
    public static int dropIndex(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        return db.dropIndex(idxName) ? 0 : -2;
    }

    /**
     * Adjusts the parallel scan chunk size (granularity) for multi-threaded search
     * dispatching.
     * Larger chunk sizes decrease thread coordination overhead, whereas smaller
     * chunk sizes
     * enable better load balancing for high core-count CPUs.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target FlatIndex
     * @param chunkSize number of records to process per parallel thread worker
     *                  chunk
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         -3 if index type is not FlatIndex, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_set_chunk_size")
    public static int setChunkSize(IsolateThread thread, CCharPointer indexName, long chunkSize) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex) {
                ((FlatIndex) index).setChunkSize(chunkSize);
                return 0;
            }
            return -3;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Sets the dynamic target energy budget threshold (tau) for early-exit tier
     * truncation.
     * Budgets are mapped to cumulative spectral energy computed from the LoRA
     * adaptation weights.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target FlatIndex
     * @param tau       target cumulative variance representation budget (e.g., 0.90
     *                  to use active tiers capturing 90% variance)
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         -3 if index type is not FlatIndex, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_set_energy_budget")
    public static int setEnergyBudget(IsolateThread thread, CCharPointer indexName, double tau) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex) {
                ((FlatIndex) index).setTargetEnergyBudget(tau);
                return 0;
            }
            return -3;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * <h3>Hardware Acceleration & DMA Direct Access Endpoint</h3>
     * Retrieves the raw off-heap virtual memory address and length of a specific
     * index tier.
     *
     * <p>
     * <b>FPGA/GPU Developers:</b> This endpoint allows custom hardware kernel logic
     * to bypass the
     * JVM execution completely. The returned address points directly to the
     * off-heap {@code mmap}ed
     * memory region. This memory layout is contiguous, read-only, cache-aligned,
     * and safe to read
     * concurrently via DMA.
     *
     * @param thread     the GraalVM isolate thread context
     * @param indexName  C string identifying the target loaded index
     * @param tierIdx    index of the target tier (0-indexed) to inspect
     * @param outAddress output pointer populated with the virtual memory address of
     *                   the tier buffer
     * @param outLength  output pointer populated with the byte size of the tier
     *                   buffer
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         -3 if invalid tier index, or -6 if the index layout is unsupported
     */
    @CEntryPoint(name = "vdb_get_tier_address")
    public static int getTierAddress(IsolateThread thread, CCharPointer indexName, int tierIdx,
            CLongPointer outAddress, CLongPointer outLength) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex) {
                FlatIndex flatIdx = (FlatIndex) index;
                long addr = flatIdx.getTierAddress(tierIdx);
                long len = flatIdx.getTierByteSize(tierIdx);
                if (addr == 0) {
                    return -3; // Invalid tier index
                }
                outAddress.write(0, addr);
                outLength.write(0, len);
                return 0;
            }
            return -6; // Unsupported index type
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Retrieves the raw off-heap virtual memory address and length of the metadata
     * (validity/tombstone) column segment for a specific loaded index.
     *
     * @param thread     the GraalVM isolate thread context
     * @param indexName  C string identifying the target loaded index
     * @param outAddress output pointer populated with the virtual memory address of
     *                   the metadata buffer
     * @param outLength  output pointer populated with the byte size of the metadata
     *                   buffer
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -6 if the index layout is unsupported
     */
    @CEntryPoint(name = "vdb_get_metadata_address")
    public static int getMetadataAddress(IsolateThread thread, CCharPointer indexName,
            CLongPointer outAddress, CLongPointer outLength) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex) {
                FlatIndex flatIdx = (FlatIndex) index;
                long addr = flatIdx.getMetadataAddress();
                long len = flatIdx.getMetadataByteSize();
                outAddress.write(0, addr);
                outLength.write(0, len);
                return 0;
            }
            return -6; // Unsupported index type
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Retrieves the raw off-heap virtual memory address and length of the record ID
     * mapping column segment for a specific loaded index.
     *
     * @param thread     the GraalVM isolate thread context
     * @param indexName  C string identifying the target loaded index
     * @param outAddress output pointer populated with the virtual memory address of
     *                   the IDs buffer
     * @param outLength  output pointer populated with the byte size of the IDs
     *                   buffer
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -6 if the index layout is unsupported
     */
    @CEntryPoint(name = "vdb_get_ids_address")
    public static int getIdsAddress(IsolateThread thread, CCharPointer indexName,
            CLongPointer outAddress, CLongPointer outLength) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex) {
                FlatIndex flatIdx = (FlatIndex) index;
                long addr = flatIdx.getIdsAddress();
                long len = flatIdx.getIdsByteSize();
                outAddress.write(0, addr);
                outLength.write(0, len);
                return 0;
            }
            return -6; // Unsupported index type
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Binarizes a single float vector using the index's Rademacher signs
     * preconditioning and
     * Walsh-Hadamard Transform rotation block operators.
     * The output packed array pointer must be pre-allocated by the caller to fit
     * {@code ceil(dimension/64)} longs.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target index config (provides signs
     *                  and dimensions)
     * @param inVector  C float array pointer storing the input vector of size
     *                  {@code (dimension)}
     * @param outPacked C long array pointer of size {@code (dimension/64)}
     *                  populated with binarized bits
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_transform_and_quantize")
    public static int transformAndQuantize(IsolateThread thread, CCharPointer indexName, CFloatPointer inVector,
            CLongPointer outPacked) {
        if (db == null) {
            return -1;
        }
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) {
                return -2;
            }
            if (index instanceof FlatIndex) {
                FlatIndex flatIdx = (FlatIndex) index;
                int dim = flatIdx.getDimension();
                float[] javaVector = new float[dim];
                for (int j = 0; j < dim; j++) {
                    javaVector[j] = inVector.read(j);
                }
                long[] packed = flatIdx.getTransformOperator().transformAndQuantize(javaVector);
                for (int i = 0; i < packed.length; i++) {
                    outPacked.write(i, packed[i]);
                }
                return 0;
            }
            return -6;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Closes the global database coordinator, freeing all resources and
     * memory-mapped virtual allocations.
     *
     * @param thread the GraalVM isolate thread context
     * @return 0 on success, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_close")
    public static int closeDb(IsolateThread thread) {
        if (db != null) {
            try {
                db.close();
                db = null;
            } catch (Throwable t) {
                t.printStackTrace();
                return -4;
            }
        }
        return 0;
    }

    // =========================================================================
    // LSM Delta Buffer C-API
    // =========================================================================

    /**
     * Creates an in-memory writable delta buffer for the named index.
     * The delta buffer enables real-time vector inserts without rewriting the
     * immutable base index files.
     *
     * @param thread         the GraalVM isolate thread context
     * @param indexName      C string identifying the index to attach the buffer to
     * @param flushThreshold soft limit on live entries. Use {@link #needsFlush} to
     *                       query recommendation
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_create_delta_buffer")
    public static int createDeltaBuffer(IsolateThread thread, CCharPointer indexName, int flushThreshold) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            db.createDeltaBuffer(idxName, flushThreshold);
            return 0;
        } catch (IllegalArgumentException e) {
            return -2;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Inserts a single float vector into the delta buffer.
     * Inserts are performed off-heap in memory with low latency.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the index to insert into
     * @param id        unique identifier of the vector record to insert
     * @param vector    C float array pointer storing the vector representation of
     *                  size {@code (dimension)}
     * @return 0 on success, -1 if database not initialized, -2 if no delta buffer
     *         is registered for this index, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_insert")
    public static int insert(IsolateThread thread, CCharPointer indexName, long id,
            CFloatPointer vector) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null)
                return -2;
            int dim = index.getDimension();
            float[] javaVector = new float[dim];
            for (int i = 0; i < dim; i++) {
                javaVector[i] = vector.read(i);
            }
            boolean ok = db.insertIntoDelta(idxName, id, javaVector);
            return ok ? 0 : -2;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Soft-deletes a record from the delta buffer (tombstone).
     * Any matching query results will filter this ID.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target index
     * @param id        unique identifier of the vector record to delete
     * @return 1 if successfully tombstoned, 0 if ID was not found, -1 if database
     *         not initialized, -2 if no delta buffer is registered
     */
    @CEntryPoint(name = "vdb_delete_from_delta")
    public static int deleteFromDelta(IsolateThread thread, CCharPointer indexName, long id) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            if (db.getDeltaBuffer(idxName) == null)
                return -2;
            return db.deleteFromDelta(idxName, id) ? 1 : 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Returns the total count of active (non-tombstoned) records in the delta
     * buffer.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target index
     * @return count of active records, -1 if database not initialized, or -2 if no
     *         delta buffer is registered
     */
    @CEntryPoint(name = "vdb_delta_size")
    public static long deltaSize(IsolateThread thread, CCharPointer indexName) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            DeltaBuffer buf = db.getDeltaBuffer(idxName);
            if (buf == null)
                return -2;
            return buf.liveSize();
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Checks if the active record count in the delta buffer exceeds the flush
     * threshold recommendation.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target index
     * @return 1 if flush is recommended, 0 if not recommended, -1 if database not
     *         initialized, or -2 if no delta buffer is registered
     */
    @CEntryPoint(name = "vdb_needs_flush")
    public static int needsFlush(IsolateThread thread, CCharPointer indexName) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            DeltaBuffer buf = db.getDeltaBuffer(idxName);
            if (buf == null)
                return -2;
            return buf.needsFlush() ? 1 : 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Performs a unified search querying both the memory-mapped base index and the
     * delta buffer,
     * merging and returning the combined top-K results.
     *
     * @param thread       the GraalVM isolate thread context
     * @param indexName    C string identifying the target loaded index
     * @param query        C float array pointer storing the query vector of size
     *                     {@code (dimension)}
     * @param k            count of nearest neighbors to retrieve (top-K)
     * @param outIds       pre-allocated C long array pointer of size {@code (k)} to
     *                     store output record IDs
     * @param outDistances pre-allocated C int array pointer of size {@code (k)} to
     *                     store output metric distances (scaled by 1,000,000)
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_search_merged")
    public static int searchMerged(IsolateThread thread, CCharPointer indexName,
            CFloatPointer query, int k,
            CLongPointer outIds, CIntPointer outDistances) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null)
                return -2;

            int dim = index.getDimension();
            float[] javaQuery = new float[dim];
            for (int i = 0; i < dim; i++) {
                javaQuery[i] = query.read(i);
            }

            List<Index.SearchResult> results = db.searchMerged(idxName, javaQuery, k);
            int count = results.size();
            for (int i = 0; i < k; i++) {
                if (i < count) {
                    outIds.write(i, results.get(i).id());
                    outDistances.write(i, results.get(i).score());
                } else {
                    outIds.write(i, -1L);
                    outDistances.write(i, Integer.MAX_VALUE);
                }
            }
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Serializes all active entries of a delta buffer to a binary backup file.
     * Output format stores vector representations in raw float32 layout.
     *
     * @param thread    the GraalVM isolate thread context
     * @param indexName C string identifying the target loaded index
     * @param path      C string destination filepath for the backup file
     * @return 0 on success, -1 if database not initialized, -2 if no delta buffer
     *         is registered, -5 on File I/O error, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_backup_delta")
    public static int backupDelta(IsolateThread thread, CCharPointer indexName, CCharPointer path) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            String filePath = CTypeConversion.toJavaString(path);
            db.backupDelta(idxName, filePath);
            return 0;
        } catch (IllegalStateException e) {
            return -2;
        } catch (java.io.IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Restores a delta buffer from a previously serialized binary file, replacing
     * any active delta buffer.
     *
     * @param thread         the GraalVM isolate thread context
     * @param indexName      C string identifying the index
     * @param path           C string filepath of the backup file to restore
     * @param flushThreshold soft limit on live entries for the restored buffer
     * @return 0 on success, -1 if database not initialized, -2 if index not found,
     *         -5 on File I/O error, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_restore_delta")
    public static int restoreDelta(IsolateThread thread, CCharPointer indexName,
            CCharPointer path, int flushThreshold) {
        if (db == null)
            return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            if (db.getIndex(idxName) == null)
                return -2;
            String filePath = CTypeConversion.toJavaString(path);
            db.restoreDelta(idxName, filePath, flushThreshold);
            return 0;
        } catch (java.io.IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    // =========================================================================
    // CUDA Acceleration C-API
    // =========================================================================

    @CEntryPoint(name = "vdb_cuda_init")
    public static int cudaInit(IsolateThread thread, int deviceId) {
        if (db == null) return -1;
        try {
            return db.cudaInit(deviceId);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    @CEntryPoint(name = "vdb_cuda_shutdown")
    public static int cudaShutdown(IsolateThread thread) {
        if (db == null) return -1;
        try {
            db.cudaShutdown();
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    @CEntryPoint(name = "vdb_cuda_is_available")
    public static int cudaIsAvailable(IsolateThread thread) {
        if (db == null) return 0;
        return db.cudaIsAvailable() ? 1 : 0;
    }

    @CEntryPoint(name = "vdb_cuda_batch_search")
    public static int cudaBatchSearch(IsolateThread thread, CCharPointer indexName, CFloatPointer queries, int numQueries,
            int k, CLongPointer outIds, CIntPointer outDistances) {
        if (db == null) return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) return -2;

            int dim = index.getDimension();
            float[][] javaQueries = new float[numQueries][dim];
            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
            }

            List<Index.SearchResult>[] results = db.cudaBatchSearch(idxName, javaQueries, k);

            for (int q = 0; q < numQueries; q++) {
                List<Index.SearchResult> queryResults = results[q];
                int count = queryResults.size();
                for (int i = 0; i < k; i++) {
                    long outId = -1;
                    int outDist = Integer.MAX_VALUE;
                    if (i < count) {
                        Index.SearchResult r = queryResults.get(i);
                        outId = r.id();
                        outDist = r.score();
                    }
                    outIds.write(q * k + i, outId);
                    outDistances.write(q * k + i, outDist);
                }
            }
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    @CEntryPoint(name = "vdb_cuda_query_planetary_grid")
    public static long cudaQueryPlanetaryGrid(IsolateThread thread, CCharPointer indexName, CFloatPointer queries,
            CIntPointer queryFamilies, CIntPointer queryThresholds, int numQueries,
            CCharPointer votingMask) {
        if (db == null) return -1;
        try {
            String idxName = CTypeConversion.toJavaString(indexName);
            Index index = db.getIndex(idxName);
            if (index == null) return -2;

            int dim = index.getDimension();
            long totalTiles = index.size();

            float[][] javaQueries = new float[numQueries][dim];
            int[] javaFamilies = new int[numQueries];
            int[] javaThresholds = new int[numQueries];

            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < dim; j++) {
                    javaQueries[q][j] = queries.read(q * dim + j);
                }
                javaFamilies[q] = queryFamilies.read(q);
                javaThresholds[q] = queryThresholds.read(q);
            }

            long rawAddress = votingMask.rawValue();
            MemorySegment maskSegment = MemorySegment.ofAddress(rawAddress).reinterpret(totalTiles);

            return db.cudaQueryPlanetaryGrid(idxName, javaQueries, javaFamilies, javaThresholds, maskSegment);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Compacts multiple compiled indexes into a single consolidated index.
     *
     * @param thread            the GraalVM isolate thread context
     * @param sourcePathsJoined semicolon-separated list of source index paths
     * @param targetPath        destination base path for consolidated index
     * @return 0 on success, -5 on File I/O error, or -4 on internal exception
     */
    @CEntryPoint(name = "vdb_compact_indexes")
    public static int compactIndexes(IsolateThread thread, CCharPointer sourcePathsJoined, CCharPointer targetPath) {
        try {
            String javaSourcePathsJoined = CTypeConversion.toJavaString(sourcePathsJoined);
            String javaTargetPath = CTypeConversion.toJavaString(targetPath);
            VectorDb.compactIndexes(javaSourcePathsJoined, javaTargetPath);
            return 0;
        } catch (IOException e) {
            return -5;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }
}
