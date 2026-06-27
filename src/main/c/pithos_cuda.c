#include "pithos_cuda.h"
#include "../cuda/pithos_kernels.h"

#include <stdio.h>

static cudaStream_t default_stream = 0;

int pithos_cuda_init(int deviceId) {
    return cuda_init_device(deviceId);
}

int pithos_cuda_shutdown() {
    return cuda_shutdown_device();
}

int pithos_cuda_is_available() {
    return cuda_is_available();
}

int pithos_cuda_get_device_count() {
    return cuda_get_device_count();
}

int pithos_cuda_alloc_pinned(void** ptr, size_t size) {
    return cuda_alloc_pinned(ptr, size);
}

int pithos_cuda_free_pinned(void* ptr) {
    return cuda_free_pinned(ptr);
}

int pithos_cuda_alloc_device(void** ptr, size_t size) {
    return cuda_alloc_device(ptr, size);
}

int pithos_cuda_free_device(void* ptr) {
    return cuda_free_device(ptr);
}

int pithos_cuda_copy_to_device(void* dst, void* src, size_t size) {
    if (!default_stream) {
        cuda_create_stream(&default_stream);
    }
    int result = cuda_copy_to_device_async(dst, src, size, default_stream);
    if (result != 0) {
        return result;
    }
    return cuda_stream_synchronize(default_stream);
}

int pithos_cuda_copy_from_device(void* dst, void* src, size_t size) {
    if (!default_stream) {
        cuda_create_stream(&default_stream);
    }
    int result = cuda_copy_from_device_async(dst, src, size, default_stream);
    if (result != 0) {
        return result;
    }
    return cuda_stream_synchronize(default_stream);
}

int pithos_cuda_launch_batch_hamming(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes
) {
    if (!default_stream) {
        cuda_create_stream(&default_stream);
    }
    
    return launch_batch_hamming_kernel(
        db_vectors, query_vectors, distances,
        num_db_vectors, num_queries, num_tiers,
        tier_offsets, tier_sizes, default_stream
    );
}

int pithos_cuda_launch_voting(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    const int* families,
    const int* thresholds,
    uint8_t* voting_mask,
    int num_db_vectors,
    int num_queries,
    int num_families,
    int num_words_per_vector
) {
    if (!default_stream) {
        cuda_create_stream(&default_stream);
    }
    
    return launch_multi_family_voting_kernel(
        db_vectors, query_vectors, families, thresholds, voting_mask,
        num_db_vectors, num_queries, num_families, num_words_per_vector, default_stream
    );
}
