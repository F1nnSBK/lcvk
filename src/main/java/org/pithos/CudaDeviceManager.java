package org.pithos;

import org.graalvm.nativeimage.c.function.CFunction;
import org.graalvm.nativeimage.c.struct.CField;
import org.graalvm.nativeimage.c.struct.CStruct;
import org.graalvm.nativeimage.c.struct.SizeOf;
import org.graalvm.word.PointerBase;

public class CudaDeviceManager {

    @CStruct("cudaDeviceProp")
    public interface CudaDeviceProperties extends PointerBase {
        @CField("totalGlobalMem") long totalGlobalMem();
        @CField("sharedMemPerBlock") long sharedMemPerBlock();
        @CField("regsPerBlock") int regsPerBlock();
        @CField("warpSize") int warpSize();
        @CField("maxThreadsPerBlock") int maxThreadsPerBlock();
        @CField("clockRate") int clockRate();
        @CField("totalConstMem") long totalConstMem();
        @CField("major") int major();
        @CField("minor") int minor();
        @CField("memoryClockRate") int memoryClockRate();
        @CField("memoryBusWidth") int memoryBusWidth();
        @CField("multiprocessorCount") int multiprocessorCount();
    }

    public static native int initialize(int deviceId);
    public static native int shutdown();
    public static native int isAvailable();
    public static native int getDeviceCount();
    public static native CudaDeviceProperties getDeviceProperties(int deviceId);

    @CStruct("cudaPointerAttributes")
    public interface CudaPointerAttributes extends PointerBase {
        @CField("type") int type();
        @CField("device") int device();
        @CField("devicePointer") long devicePointer();
        @CField("hostPointer") long hostPointer();
    }
}
