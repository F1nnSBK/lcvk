#include "pithos_cuda.h"
#include <jni.h>

JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_allocPinned(
    JNIEnv* env, 
    jclass clazz,
    jlong size
) {
    void* ptr = NULL;
    int result = pithos_cuda_alloc_pinned(&ptr, (size_t)size);
    return result == 0 ? (jlong)ptr : 0;
}

JNIEXPORT void JNICALL Java_org_pithos_CudaMemoryManager_freePinned(
    JNIEnv* env, 
    jclass clazz,
    jlong pointer
) {
    if (pointer != 0) {
        pithos_cuda_free_pinned((void*)pointer);
    }
}

JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_allocDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong size
) {
    void* ptr = NULL;
    int result = pithos_cuda_alloc_device(&ptr, (size_t)size);
    return result == 0 ? (jlong)ptr : 0;
}

JNIEXPORT void JNICALL Java_org_pithos_CudaMemoryManager_freeDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong pointer
) {
    if (pointer != 0) {
        pithos_cuda_free_device((void*)pointer);
    }
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyToDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size
) {
    return pithos_cuda_copy_to_device((void*)dst, (void*)src, (size_t)size);
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyFromDevice(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size
) {
    return pithos_cuda_copy_from_device((void*)dst, (void*)src, (size_t)size);
}

JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_createStream(
    JNIEnv* env, 
    jclass clazz
) {
    cudaStream_t* stream = malloc(sizeof(cudaStream_t));
    if (cudaStreamCreate(stream) != cudaSuccess) {
        free(stream);
        return 0;
    }
    return (jlong)stream;
}

JNIEXPORT void JNICALL Java_org_pithos_CudaMemoryManager_destroyStream(
    JNIEnv* env, 
    jclass clazz,
    jlong stream
) {
    if (stream != 0) {
        cudaStreamDestroy(*(cudaStream_t*)stream);
        free((void*)stream);
    }
}

JNIEXPORT jlong JNICALL Java_org_pithos_CudaMemoryManager_getDirectBufferAddress(
    JNIEnv* env, 
    jclass clazz,
    jobject buffer
) {
    return (jlong)(*env)->GetDirectBufferAddress(env, buffer);
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyToDeviceAsync(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size,
    jlong stream
) {
    return pithos_cuda_copy_to_device_async((void*)dst, (void*)src, (size_t)size, (cudaStream_t*)stream);
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_copyFromDeviceAsync(
    JNIEnv* env, 
    jclass clazz,
    jlong dst,
    jlong src,
    jlong size,
    jlong stream
) {
    return pithos_cuda_copy_from_device_async((void*)dst, (void*)src, (size_t)size, (cudaStream_t*)stream);
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaMemoryManager_streamSynchronize(
    JNIEnv* env, 
    jclass clazz,
    jlong stream
) {
    return pithos_cuda_stream_synchronize((cudaStream_t*)stream);
}