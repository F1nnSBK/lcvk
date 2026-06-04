#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lunar_core.h"

// Helper function to calculate elapsed time in milliseconds
double get_elapsed_ms(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 + (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// Helper function to calculate elapsed time in microseconds
double get_elapsed_us(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000000.0 + (double)(end.tv_nsec - start.tv_nsec) / 1000.0;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    graal_isolate_t *isolate = NULL;
    graal_isolatethread_t *thread = NULL;
    struct timespec start, end;
    
    double t_isolate_create = 0.0;
    double t_db_init = 0.0;
    double t_compile_index = 0.0;
    double t_load_index = 0.0;
    double t_knn_search = 0.0;
    double t_voting_search = 0.0;
    double t_isolate_teardown = 0.0;

    printf("[C Client] Creating GraalVM Isolate...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (graal_create_isolate(NULL, &isolate, &thread) != 0) {
        fprintf(stderr, "[C Client] Failed to create GraalVM isolate!\n");
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_isolate_create = get_elapsed_ms(start, end);
    printf("[C Client] Isolate created in %.3f ms\n", t_isolate_create);

    printf("[C Client] Initializing LCVK Database...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    int status = vdb_init(thread);
    if (status != 0) {
        fprintf(stderr, "[C Client] Failed to initialize database (code: %d)!\n", status);
        graal_tear_down_isolate(thread);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_db_init = get_elapsed_ms(start, end);
    printf("[C Client] Database initialized in %.3f ms\n", t_db_init);

    // 1. Compile test database file (PLAN header, 384-bit binary vectors)
    printf("[C Client] Compiling test binary vector file 'lunar_test.lcvk'...\n");
    long long ids[] = {0, 1, 2};
    // 3 vectors, each 6 longs (384 bits)
    long long vectors[] = {
        // Record 0: all zeros
        0, 0, 0, 0, 0, 0,
        // Record 1: 4 bits set in first long (value 15 = 0b1111)
        15, 0, 0, 0, 0, 0,
        // Record 2: 2 bits set in first long (value 3 = 0b0011)
        3, 0, 0, 0, 0, 0
    };

    clock_gettime(CLOCK_MONOTONIC, &start);
    // Planet ID = 1 (Moon), Radius = 1737400 meters
    status = vdb_compile_index_file(thread, "lunar_test.lcvk", (char)1, 1737400LL, ids, vectors, 3);
    if (status != 0) {
        fprintf(stderr, "[C Client] Index compilation failed (code: %d)\n", status);
        graal_tear_down_isolate(thread);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_compile_index = get_elapsed_ms(start, end);
    printf("[C Client] Index compiled in %.3f ms\n", t_compile_index);

    // 2. Load the compiled index using off-heap memory mapping
    printf("[C Client] Loading memory-mapped index 'lunar_index' from 'lunar_test.lcvk'...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    status = vdb_load_index(thread, "lunar_index", "lunar_test.lcvk");
    if (status != 0) {
        fprintf(stderr, "[C Client] Failed to load index (code: %d)\n", status);
        graal_tear_down_isolate(thread);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_load_index = get_elapsed_ms(start, end);
    printf("[C Client] Mapped index loaded in %.3f ms\n", t_load_index);

    long long size = vdb_size(thread, "lunar_index");
    printf("[C Client] Mapped index size: %lld records\n", size);

    // 3. Test Standard Batch Search (KNN)
    printf("\n[C Client] Running standard KNN batch search...\n");
    long long knn_queries[] = {
        // Query 1: matches Record 1 perfectly (15)
        15, 0, 0, 0, 0, 0,
        // Query 2: close to Record 2 (3), query with 2 (0b0010)
        2, 0, 0, 0, 0, 0
    };
    int num_queries = 2;
    int k = 2;

    long long out_ids[4];
    int out_distances[4];

    clock_gettime(CLOCK_MONOTONIC, &start);
    status = vdb_batch_search(thread, "lunar_index", knn_queries, num_queries, k, out_ids, out_distances);
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_knn_search = get_elapsed_us(start, end);
    
    if (status != 0) {
        fprintf(stderr, "[C Client] Batch search failed (code: %d)\n", status);
    } else {
        printf("[C Client] KNN search completed in %.3f us (%.4f ms). Results:\n", t_knn_search, t_knn_search / 1000.0);
        for (int q = 0; q < num_queries; q++) {
            printf("  Query %d results:\n", q + 1);
            for (int i = 0; i < k; i++) {
                int idx = q * k + i;
                printf("    Rank %d: Vector ID = %lld, Hamming Distance = %d\n", i + 1, out_ids[idx], out_distances[idx]);
            }
        }
    }

    // 4. Test Multi-Family Resonant Voting
    printf("\n[C Client] Running Multi-Family Resonant Voting search...\n");
    long long voting_queries[] = {
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0,
        15, 0, 0, 0, 0, 0
    };
    int voting_families[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int voting_thresholds[] = {0, 0, 0, 0, 0, 0, 0, 0}; // Strict exact match (Hamming distance = 0)
    int num_voting_queries = 8;

    char *voting_mask = (char *)malloc(size);
    memset(voting_mask, 0, size);

    clock_gettime(CLOCK_MONOTONIC, &start);
    long long resonant_count = vdb_query_planetary_grid(
        thread, "lunar_index", voting_queries, voting_families, voting_thresholds, num_voting_queries, voting_mask
    );
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_voting_search = get_elapsed_us(start, end);

    if (resonant_count < 0) {
        fprintf(stderr, "[C Client] query_planetary_grid failed (code: %lld)\n", resonant_count);
    } else {
        printf("[C Client] Resonant voting completed in %.3f us (%.4f ms). Found %lld resonant tiles (>= 7 active bits):\n", 
               t_voting_search, t_voting_search / 1000.0, resonant_count);
        for (int i = 0; i < size; i++) {
            printf("  Tile ID %d: voting_mask = 0x%02X (popcount = %d)\n", i, (unsigned char)voting_mask[i], __builtin_popcount((unsigned char)voting_mask[i]));
        }
    }

    free(voting_mask);

    printf("\n[C Client] Closing LCVK Database...\n");
    vdb_close(thread);

    printf("[C Client] Tearing down GraalVM Isolate...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    graal_tear_down_isolate(thread);
    clock_gettime(CLOCK_MONOTONIC, &end);
    t_isolate_teardown = get_elapsed_ms(start, end);
    printf("[C Client] Isolate torn down in %.3f ms\n", t_isolate_teardown);

    // Beautiful High-Resolution Timing Report
    printf("\n");
    printf("========================================================================\n");
    printf("                  LCVK NATIVE ENGINE BENCHMARK REPORT                   \n");
    printf("========================================================================\n");
    printf("  Operation                       | Latency                             \n");
    printf("----------------------------------+-------------------------------------\n");
    printf("  1. GraalVM Isolate Creation     | %10.3f ms                           \n", t_isolate_create);
    printf("  2. LCVK DB Engine Init          | %10.3f ms                           \n", t_db_init);
    printf("  3. Offline Index Compilation    | %10.3f ms                           \n", t_compile_index);
    printf("  4. Zero-Copy Index Memory-Map   | %10.3f ms                           \n", t_load_index);
    printf("  5. Parallel KNN Batch Search    | %10.3f us  (Avg: %8.3f us/query)   \n", t_knn_search, t_knn_search / num_queries);
    printf("  6. Resonant Voting Search       | %10.3f us  (Avg: %8.3f us/query)   \n", t_voting_search, t_voting_search / num_voting_queries);
    printf("  7. GraalVM Isolate Teardown     | %10.3f ms                           \n", t_isolate_teardown);
    printf("========================================================================\n");
    printf("  Index Configuration: %lld records, 384-bit vectors off-heap mapped\n", size);
    printf("========================================================================\n");

    printf("[C Client] Finished successfully!\n");
    return 0;
}
