package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.*;

public class CudaKernelTest {

    private boolean isCudaSupported() {
        try {
            return CudaDeviceManager.isAvailable() != 0;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    @Test
    public void testCudaAvailability() {
        Assumptions.assumeTrue(isCudaSupported(), "CUDA not available");
        
        int deviceCount = CudaDeviceManager.getDeviceCount();
        assertTrue(deviceCount > 0, "No CUDA devices found");
    }

    @Test
    public void testCudaInitialization() {
        Assumptions.assumeTrue(isCudaSupported(), "CUDA not available");
        
        int result = CudaDeviceManager.initialize(0);
        assertEquals(0, result, "CUDA initialization failed");
        
        CudaDeviceManager.shutdown();
    }

    @Test
    public void testMemoryAllocation() {
        Assumptions.assumeTrue(isCudaSupported(), "CUDA not available");
        
        CudaDeviceManager.initialize(0);
        
        try {
            long size = 1024 * 1024;
            
            long pinnedPtr = CudaMemoryManager.allocPinned(size);
            assertTrue(pinnedPtr != 0, "Pinned memory allocation failed");
            
            long devicePtr = CudaMemoryManager.allocDevice(size);
            assertTrue(devicePtr != 0, "Device memory allocation failed");
            
            CudaMemoryManager.freePinned(pinnedPtr);
            CudaMemoryManager.freeDevice(devicePtr);
        } finally {
            CudaDeviceManager.shutdown();
        }
    }

    @Test
    public void testMemoryTransfer() {
        Assumptions.assumeTrue(isCudaSupported(), "CUDA not available");
        
        CudaDeviceManager.initialize(0);
        
        try {
            int size = 256 * 4;
            ByteBuffer hostBuffer = ByteBuffer.allocateDirect(size);
            
            for (int i = 0; i < 256; i++) {
                hostBuffer.putFloat(i);
            }
            hostBuffer.rewind();
            
            long devicePtr = CudaMemoryManager.allocDevice(size);
            assertTrue(devicePtr != 0, "Device memory allocation failed");
            
            long hostBufferPtr = CudaMemoryManager.getDirectBufferAddress(hostBuffer);
            int result = CudaMemoryManager.copyToDevice(devicePtr, hostBufferPtr, size);
            assertEquals(0, result, "Memory transfer to device failed");
            
            ByteBuffer resultBuffer = ByteBuffer.allocateDirect(size);
            long resultBufferPtr = CudaMemoryManager.getDirectBufferAddress(resultBuffer);
            result = CudaMemoryManager.copyFromDevice(resultBufferPtr, devicePtr, size);
            assertEquals(0, result, "Memory transfer from device failed");
            
            resultBuffer.rewind();
            for (int i = 0; i < 256; i++) {
                assertEquals(i, (int) resultBuffer.getFloat(), "Data mismatch at index " + i);
            }
            
            CudaMemoryManager.freeDevice(devicePtr);
        } finally {
            CudaDeviceManager.shutdown();
        }
    }

    @Test
    public void testStreamCreation() {
        Assumptions.assumeTrue(isCudaSupported(), "CUDA not available");
        
        CudaDeviceManager.initialize(0);
        
        try {
            long stream = CudaMemoryManager.createStream();
            assertTrue(stream != 0, "Stream creation failed");
            
            CudaMemoryManager.destroyStream(stream);
        } finally {
            CudaDeviceManager.shutdown();
        }
    }
}
