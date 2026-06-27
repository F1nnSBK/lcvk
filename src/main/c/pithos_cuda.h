#ifndef PITHOS_CUDA_H
#define PITHOS_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int pithos_cuda_init(int deviceId);
int pithos_cuda_shutdown();
int pithos_cuda_is_available();
int pithos_cuda_get_device_count();

int pithos_cuda_alloc_pinned(void** ptr, size_t size);
int pithos_cuda_free_pinned(void* ptr);
int pithos_cuda_alloc_device(void** ptr, size_t size);
int pithos_cuda_free_device(void* ptr);

int pithos_cuda_copy_to_device(void* dst, void* src, size_t size);
int pithos_cuda_copy_from_device(void* dst, void* src, size_t size);

int pithos_cuda_launch_batch_hamming(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes
);

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
);

// Async operations
int pithos_cuda_copy_to_device_async(void* dst, void* src, size_t size, void* stream);
int pithos_cuda_copy_from_device_async(void* dst, void* src, size_t size, void* stream);
int pithos_cuda_stream_synchronize(void* stream);

// Stream management
int cuda_create_stream(void** stream);

#ifdef __cplusplus
}
#endif

#endif
