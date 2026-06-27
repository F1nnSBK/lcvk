#include "pithos_cuda.h"
#include <string.h>

JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_initialize(
    JNIEnv* env, 
    jclass clazz,
    jint deviceId
) {
    return pithos_cuda_init(deviceId);
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_shutdown(
    JNIEnv* env, 
    jclass clazz
) {
    return pithos_cuda_shutdown();
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_isAvailable(
    JNIEnv* env, 
    jclass clazz
) {
    return pithos_cuda_is_available();
}

JNIEXPORT jint JNICALL Java_org_pithos_CudaDeviceManager_getDeviceCount(
    JNIEnv* env, 
    jclass clazz
) {
    return pithos_cuda_get_device_count();
}

JNIEXPORT jobject JNICALL Java_org_pithos_CudaDeviceManager_getDeviceProperties(
    JNIEnv* env, 
    jclass clazz,
    jint deviceId
) {
    // Implementation would go here
    // For now, return null as placeholder
    return NULL;
}