#ifndef PITHOS_KERNELS_H
#define PITHOS_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_WORDS_PER_VECTOR 6
#define MAX_TIERS 8
#define MAX_FAMILIES 8

#ifdef __cplusplus
extern "C" {
#endif

int launch_batch_hamming_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_tiers,
    const int* tier_offsets,
    const int* tier_sizes,
    cudaStream_t stream
);

int launch_batch_hamming_optimized_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_words_per_vector,
    cudaStream_t stream
);

int launch_multi_family_voting_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    const int* families,
    const int* thresholds,
    uint8_t* voting_mask,
    int num_db_vectors,
    int num_queries,
    int num_families,
    int num_words_per_vector,
    cudaStream_t stream
);

int launch_walsh_hadamard_kernel(
    float* input,
    float* output,
    int num_vectors,
    int dimension,
    cudaStream_t stream
);

int cuda_alloc_pinned(void** ptr, size_t size);
int cuda_free_pinned(void* ptr);
int cuda_alloc_device(void** ptr, size_t size);
int cuda_free_device(void* ptr);
int cuda_copy_to_device_async(void* dst, void* src, size_t size, cudaStream_t stream);
int cuda_copy_from_device_async(void* dst, void* src, size_t size, cudaStream_t stream);
int cuda_stream_synchronize(cudaStream_t stream);
int cuda_create_stream(cudaStream_t* stream);
int cuda_destroy_stream(cudaStream_t stream);
int cuda_init_device(int deviceId);
int cuda_shutdown_device();
int cuda_is_available();
int cuda_get_device_count();
int cuda_get_device_properties(int deviceId, void* prop);

#ifdef __cplusplus
}
#endif

#endif
