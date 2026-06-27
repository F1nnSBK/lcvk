package org.pithos;

public class CudaNativeBindings {

    static native int pithos_cuda_launch_batch_hamming(
        long[] deviceTierBuffers, long deviceQueries, long hostDistances,
        int numDbVectors, int numQueries, int numTiers, int[] tierOffsets, int[] tierSizes
    );

    static native int pithos_cuda_launch_voting(
        long[] deviceTierBuffers, long deviceQueries, long deviceFamilies, long deviceThresholds,
        long deviceVotingMask, int numDbVectors, int numQueries, int numFamilies, int numWordsPerVector
    );
}
