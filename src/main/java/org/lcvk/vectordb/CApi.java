package org.lcvk.vectordb;

import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CIntPointer;
import org.graalvm.nativeimage.c.type.CLongPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * C-API entry points for the binary vector database (LCVK).
 * Exposes methods to native callers using GraalVM {@link CEntryPoint}.
 */
public class CApi {
    private static VectorDb db;

    private CApi() {}

    /**
     * Initialisiert den globalen Datenbank-Koordinator.
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
     * Mappt eine existierende Datenbankdatei off-heap in den virtuellen Speicher.
     *
     * @param name C-String Registername des Index
     * @param path C-String Dateipfad zur Datenbankdatei
     */
    @CEntryPoint(name = "vdb_load_index")
    public static int loadIndex(IsolateThread thread, CCharPointer name, CCharPointer path) {
        if (db == null) {
            return -1;
        }
        try {
            String indexName = CTypeConversion.toJavaString(name);
            String filePath = CTypeConversion.toJavaString(path);
            db.loadIndex(indexName, filePath);
            return 0;
        } catch (IOException e) {
            return -5; // FILE IO ERROR
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Fuehrt eine batchierte KNN-Suche aus.
     *
     * @param indexName    Name des Index
     * @param queries      Zeiger auf flaches Array mit Suchvektoren (numQueries * 6 longs)
     * @param numQueries   Anzahl der Queries
     * @param k            Anzahl der nächsten Nachbarn pro Query
     * @param outIds       C-Array der Groesse (numQueries * k), in das IDs geschrieben werden
     * @param outDistances C-Array der Groesse (numQueries * k), in das Distanzen geschrieben werden
     */
    @CEntryPoint(name = "vdb_batch_search")
    public static int batchSearch(IsolateThread thread, CCharPointer indexName, CLongPointer queries, int numQueries, int k,
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

            // Kopiere Queries in Java-Struktur
            long[][] javaQueries = new long[numQueries][6];
            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < 6; j++) {
                    javaQueries[q][j] = queries.read(q * 6 + j);
                }
            }

            // Ausfuehren der parallelen Suche
            List<Index.SearchResult>[] results = index.batchSearch(javaQueries, k);

            // Schreibe Ergebnisse direkt in die C-Array-Buffer
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
            return 0; // SUCCESS
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Fuehrt eine batchierte Suche aus und fuehrt ein Multi-Family Bitmask Voting ueber den globalen Voting-Buffer aus.
     * Schreibt das Voting-Ergebnis (Bits 0-7) direkt zero-copy in den vom Aufrufer bereitgestellten Buffer.
     *
     * @param indexName       Name des Index
     * @param queries         Zeiger auf flaches Array mit Suchvektoren (numQueries * 6 longs)
     * @param queryFamilies   Zeiger auf Array mit Familie-ID pro Query (Länge numQueries)
     * @param queryThresholds Zeiger auf Array mit Hamming-Distanztoleranz pro Query (Länge numQueries)
     * @param numQueries      Anzahl der Queries
     * @param votingMask      C-Pointer auf pre-allokiertes Byte-Array (Länge = totalTiles)
     * @return Anzahl der resonant-gematchten Tiles (>= 7 gesetzte Bits), oder Fehlercode bei Wert < 0
     */
    @CEntryPoint(name = "vdb_query_planetary_grid")
    public static long queryPlanetaryGrid(IsolateThread thread, CCharPointer indexName, CLongPointer queries,
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

            long totalTiles = index.size();

            // Kopiere FFI-Arrays in native Java-Typen
            long[][] javaQueries = new long[numQueries][6];
            int[] javaFamilies = new int[numQueries];
            int[] javaThresholds = new int[numQueries];

            for (int q = 0; q < numQueries; q++) {
                for (int j = 0; j < 6; j++) {
                    javaQueries[q][j] = queries.read(q * 6 + j);
                }
                javaFamilies[q] = queryFamilies.read(q);
                javaThresholds[q] = queryThresholds.read(q);
            }

            // Wickle den rohen C-Buffer zero-copy in ein Panama MemorySegment
            long rawAddress = votingMask.rawValue();
            MemorySegment maskSegment = MemorySegment.ofAddress(rawAddress).reinterpret(totalTiles);

            // Ausfuehren des Multi-Query Scans mit Bitmask Voting
            return index.queryPlanetaryGrid(javaQueries, javaFamilies, javaThresholds, maskSegment);
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Kompiliert rohe Vektordaten offline direkt in eine optimierte Binärdatei mit PLAN-Header.
     */
    @CEntryPoint(name = "vdb_compile_index_file")
    public static int compileIndexFile(IsolateThread thread, CCharPointer path, byte planetId, long planetRadius,
                                       CLongPointer ids, CLongPointer vectors, int numRecords) {
        try {
            String filePath = CTypeConversion.toJavaString(path);
            List<VectorRecord> records = new ArrayList<>(numRecords);
            for (int i = 0; i < numRecords; i++) {
                long id = ids.read(i);
                long[] vector = new long[6];
                for (int j = 0; j < 6; j++) {
                    vector[j] = vectors.read(i * 6 + j);
                }
                records.add(new VectorRecord(id, vector));
            }
            VectorDb.compileIndexFile(filePath, planetId, planetRadius, records);
            return 0;
        } catch (Throwable t) {
            t.printStackTrace();
            return -4;
        }
    }

    /**
     * Gibt die Anzahl der Datensätze im Index zurück.
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
     * Loescht einen Index aus der Registrierung.
     */
    @CEntryPoint(name = "vdb_drop_index")
    public static int dropIndex(IsolateThread thread, CCharPointer indexName) {
        if (db == null) {
            return -1;
        }
        String idxName = CTypeConversion.toJavaString(indexName);
        return db.dropIndex(idxName) ? 0 : -2;
    }
}
