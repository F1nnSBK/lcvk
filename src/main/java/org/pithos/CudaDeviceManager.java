package org.pithos;

public class CudaDeviceManager {

    public static native int initialize(int deviceId);
    public static native int shutdown();
    public static native int isAvailable();
    public static native int getDeviceCount();
}
