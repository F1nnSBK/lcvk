package org.pithos;

import java.nio.ByteBuffer;

public class CudaMemoryManager {

    private static final int DEFAULT_STREAM_COUNT = 4;

    private final long[] streams;
    private final long pinnedBuffer;
    private final long deviceBuffer;
    private final long bufferSize;

    public CudaMemoryManager(long bufferSize) {
        this.bufferSize = bufferSize;
        this.streams = new long[DEFAULT_STREAM_COUNT];
        for (int i = 0; i < streams.length; i++) {
            streams[i] = createStream();
        }
        this.pinnedBuffer = allocPinned(bufferSize);
        this.deviceBuffer = allocDevice(bufferSize);
    }

    public static native long allocPinned(long size);
    public static native void freePinned(long pointer);
    public static native long allocDevice(long size);
    public static native void freeDevice(long pointer);
    public static native int copyToDevice(long dst, long src, long size);
    public static native int copyFromDevice(long dst, long src, long size);
    public static native long createStream();
    public static native void destroyStream(long stream);

    public void asyncTransferToDevice(ByteBuffer hostBuffer, long devicePtr, int streamIndex) {
        long size = hostBuffer.remaining();
        long hostPtr = getDirectBufferAddress(hostBuffer);
        copyToDeviceAsync(devicePtr, hostPtr, size, streams[streamIndex]);
    }

    public void asyncTransferFromDevice(long hostPtr, long devicePtr, long size, int streamIndex) {
        copyFromDeviceAsync(hostPtr, devicePtr, size, streams[streamIndex]);
    }

    public void synchronizeStream(int streamIndex) {
        streamSynchronize(streams[streamIndex]);
    }

    public long getPinnedBuffer() {
        return pinnedBuffer;
    }

    public long getDeviceBuffer() {
        return deviceBuffer;
    }

    public long getStream(int index) {
        return streams[index];
    }

    public void shutdown() {
        for (long stream : streams) {
            destroyStream(stream);
        }
        freePinned(pinnedBuffer);
        freeDevice(deviceBuffer);
    }

    private static native long getDirectBufferAddress(ByteBuffer buffer);
    private static native int copyToDeviceAsync(long dst, long src, long size, long stream);
    private static native int copyFromDeviceAsync(long dst, long src, long size, long stream);
    private static native int streamSynchronize(long stream);
}
