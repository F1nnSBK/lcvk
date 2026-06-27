package org.pithos;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.*;

public class CudaKernelTest {

    @Test
    public void testCudaAvailability() {
        Assumptions.assumeTrue(CudaDeviceManager.isAvailable() != 0, "CUDA not available");
        
        int deviceCount = CudaDeviceManager.getDeviceCount();
        assertTrue(deviceCount > 0, "No CUDA devices found");
    }

    @Test
    public void testCudaInitialization() {
        Assumptions.assumeTrue(CudaDeviceManager.isAvailable() != 0, "CUDA not available");
        
        int result = CudaDeviceManager.initialize(0);
        assertEquals(0, result, "CUDA initialization failed");
        
        CudaDeviceManager.shutdown();
    }

    @Test
    public void testMemoryAllocation() {
        Assumptions.assumeTrue(CudaDeviceManager.isAvailable() != 0, "CUDA not available");
        
        CudaDeviceManager.initialize(0);
        
        try {
            long size = 1024 * 1024; // 1MB
            
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
        Assumptions.assumeTrue(CudaDeviceManager.isAvailable() != 0, "CUDA not available");
        
        CudaDeviceManager.initialize(0);
        
        try {
            int size = 256 * 4; // 256 floats
            ByteBuffer hostBuffer = ByteBuffer.allocateDirect(size);
            
            for (int i = 0; i < 256; i++) {
                hostBuffer.putFloat(i);
            }
            hostBuffer.rewind();
            
            long devicePtr = CudaMemoryManager.allocDevice(size);
            assertTrue(devicePtr != 0, "Device memory allocation failed");
            
            int result = CudaMemoryManager.copyToDevice(devicePtr, hostBuffer, size);
            assertEquals(0, result, "Memory transfer to device failed");
            
            ByteBuffer resultBuffer = ByteBuffer.allocateDirect(size);
            result = CudaMemoryManager.copyFromDevice(resultBuffer, devicePtr, size);
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
        Assumptions.assumeTrue(CudaDeviceManager.isAvailable() != 0, "CUDA not available");
        
        CudaDeviceManager.initialize(0);
        
        try {
            long stream = CudaMemoryManager.createStream();
            assertTrue(stream != 0, "Stream creation failed");
            
            CudaMemoryManager.destroyStream(stream);
        } finally {
            CudaDeviceManager.shutdown();
        }
    }

    @Test
    public void testDeviceProperties() {
        Assumptions.assumeTrue(CudaDeviceManager.isAvailable() != 0, "CUDA not available");
        
        CudaDeviceManager.CudaDeviceProperties props = CudaDeviceManager.getDeviceProperties(0);
        assertNotNull(props, "Failed to get device properties");
        
        assertNotNull(props.name(), "Device name is null");
        assertTrue(props.totalGlobalMem() > 0, "Invalid global memory size");
        assertTrue(props.maxThreadsPerBlock() > 0, "Invalid max threads per block");
    }
}
