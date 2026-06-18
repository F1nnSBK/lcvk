#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pithos.h"

int main() {
    graal_isolate_t *isolate = NULL;
    graal_isolatethread_t *thread = NULL;

    printf("[Pithos Demo] Creating GraalVM Isolate...\n");
    if (graal_create_isolate(NULL, &isolate, &thread) != 0) {
        fprintf(stderr, "[Error] Failed to create GraalVM isolate!\n");
        return 1;
    }

    printf("[Pithos Demo] Initializing database coordinator...\n");
    if (vdb_init(thread) != 0) {
        fprintf(stderr, "[Error] Failed to initialize Pithos DB engine!\n");
        graal_tear_down_isolate(thread);
        return 1;
    }

    // Load an existing index (assuming pithos_test.bin exists)
    const char *index_name = "demo_index";
    const char *index_path = "pithos_test.bin";
    
    printf("[Pithos Demo] Attempting to load memory-mapped index '%s'...\n", index_name);
    int status = vdb_load_index(thread, (char *)index_name, (char *)index_path);
    if (status != 0) {
        printf("[Pithos Demo] Index '%s' not found. This is expected if the benchmark has not run yet.\n", index_path);
    } else {
        long long size = vdb_size(thread, (char *)index_name);
        printf("[Pithos Demo] Loaded index with %lld records successfully.\n", size);
    }

    printf("[Pithos Demo] Shutting down database and freeing mapped memory...\n");
    vdb_close(thread);
    graal_tear_down_isolate(thread);
    
    return 0;
}
