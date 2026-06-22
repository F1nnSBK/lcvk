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
 */
public class CApi {
    private static VectorDb db;

    private CApi() {}

    /**
     * Initializes the global database coordinator.
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
     * Maps an existing database off-heap in the virtual memory without custom weights.
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
            return -5; // FILE IO ERROR
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Maps an existing database off-heap, supplying frozen LoRA weights.
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
     * Retrieves database information metadata attributes.
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
     * Performs a batch KNN search on float vectors.
     */
    @CEntryPoint(name = "vdb_batch_search")
    public static int batchSearch(IsolateThread thread, CCharPointer indexName, CFloatPointer queries, int numQueries, int k,
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
     * Performs a batch search and outputs matching candidate votes into the pre-allocated voting mask.
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
     * Compiles raw float records into a multi-tier database file layout.
     */
    @CEntryPoint(name = "vdb_compile_index_file")
    public static int compileIndexFile(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
                                       int dimension, CIntPointer tiers, int numTiers,
                                       CLongPointer ids, CFloatPointer vectors, int numRecords) {
        return compileIndexFileV2(thread, path, planetId, planetRadius, dimension, tiers, numTiers, ids, vectors, numRecords, 0);
    }

    /**
     * Compiles raw float records into a multi-tier database file layout with qMode.
     */
    @CEntryPoint(name = "vdb_compile_index_file_v2")
    public static int compileIndexFileV2(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
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

    @CEntryPoint(name = "vdb_drop_index")
    public static int dropIndex(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        return db.dropIndex(idxName) ? 0 : -2;
    }

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
     * Sets dynamic target energy budget tau (e.g. 0.90) for FlatIndex.
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
}
