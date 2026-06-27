#include "pithos_cuda.h"

// CUDA C-API functions called from Java CEntryPoint methods

JNIEXPORT jint JNICALL Java_org_pithos_CApi_cudaInit(
    JNIEnv* env, 
    jobject obj, 
    jobject thread,
    jint deviceId
) {
    return pithos_cuda_init(deviceId);
}

JNIEXPORT jint JNICALL Java_org_pithos_CApi_cudaShutdown(
    JNIEnv* env, 
    jobject obj, 
    jobject thread
) {
    return pithos_cuda_shutdown();
}

JNIEXPORT jint JNICALL Java_org_pithos_CApi_cudaIsAvailable(
    JNIEnv* env, 
    jobject obj, 
    jobject thread
) {
    return pithos_cuda_is_available();
}

JNIEXPORT jint JNICALL Java_org_pithos_CApi_cudaBatchSearch(
    JNIEnv* env,
    jobject obj,
    jobject thread,
    jbyteArray indexName,
    jobject queries,
    jint numQueries,
    jint k,
    jlongArray outIds,
    jintArray outDistances
) {
    // This function bridges Java CEntryPoint to C CUDA implementation
    // For GraalVM Native Image, we use direct function calls
    
    // Convert parameters
    const char* index_name = (*env)->GetByteArrayElements(env, indexName, NULL);
    float* queries_ptr = (float*) (*env)->GetDirectBufferAddress(env, queries);
    jlong* out_ids = (*env)->GetLongArrayElements(env, outIds, NULL);
    jint* out_dists = (*env)->GetIntArrayElements(env, outDistances, NULL);
    
    int status = pithos_cuda_launch_batch_hamming(
        NULL, queries_ptr, (int*)out_dists,
        0, numQueries, 0, NULL, NULL
    );
    
    (*env)->ReleaseByteArrayElements(env, indexName, (jbyte*)index_name, JNI_ABORT);
    (*env)->ReleaseLongArrayElements(env, outIds, out_ids, 0);
    (*env)->ReleaseIntArrayElements(env, outDistances, out_dists, 0);
    
    return status;
}

JNIEXPORT jlong JNICALL Java_org_pithos_CApi_cudaQueryPlanetaryGrid(
    JNIEnv* env,
    jobject obj,
    jobject thread,
    jbyteArray indexName,
    jobject queries,
    jobject queryFamilies,
    jobject queryThresholds,
    jint numQueries,
    jobject votingMask
) {
    // Convert parameters
    const char* index_name = (*env)->GetByteArrayElements(env, indexName, NULL);
    float* queries_ptr = (float*) (*env)->GetDirectBufferAddress(env, queries);
    int* families_ptr = (int*) (*env)->GetDirectBufferAddress(env, queryFamilies);
    int* thresholds_ptr = (int*) (*env)->GetDirectBufferAddress(env, queryThresholds);
    uint8_t* mask_ptr = (uint8_t*) (*env)->GetDirectBufferAddress(env, votingMask);
    
    long result = pithos_cuda_launch_voting(
        NULL, queries_ptr, families_ptr, thresholds_ptr, mask_ptr,
        0, numQueries, 0, 0
    );
    
    (*env)->ReleaseByteArrayElements(env, indexName, (jbyte*)index_name, JNI_ABORT);
    
    return result;
}
