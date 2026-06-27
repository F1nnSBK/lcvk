#ifndef PITHOS_KERNELS_H
#define PITHOS_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_WORDS_PER_VECTOR 6
#define MAX_TIERS 8
#define MAX_FAMILIES 8

extern "C" int launch_batch_hamming_kernel(
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

extern "C" int launch_batch_hamming_optimized_kernel(
    const uint64_t* db_vectors,
    const uint64_t* query_vectors,
    int* distances,
    int num_db_vectors,
    int num_queries,
    int num_words_per_vector,
    cudaStream_t stream
);

extern "C" int launch_multi_family_voting_kernel(
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

extern "C" int launch_walsh_hadamard_kernel(
    float* input,
    float* output,
    int num_vectors,
    int dimension,
    cudaStream_t stream
);

extern "C" int cuda_alloc_pinned(void** ptr, size_t size);
extern "C" int cuda_free_pinned(void* ptr);
extern "C" int cuda_alloc_device(void** ptr, size_t size);
extern "C" int cuda_free_device(void* ptr);
extern "C" int cuda_copy_to_device_async(void* dst, void* src, size_t size, cudaStream_t stream);
extern "C" int cuda_copy_from_device_async(void* dst, void* src, size_t size, cudaStream_t stream);
extern "C" int cuda_stream_synchronize(cudaStream_t stream);
extern "C" int cuda_create_stream(cudaStream_t* stream);
extern "C" int cuda_destroy_stream(cudaStream_t stream);
extern "C" int cuda_init_device(int deviceId);
extern "C" int cuda_shutdown_device();
extern "C" int cuda_is_available();
extern "C" int cuda_get_device_count();
extern "C" int cuda_get_device_properties(int deviceId, void* prop);

#endif
