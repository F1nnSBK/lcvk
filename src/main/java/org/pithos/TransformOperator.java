package org.pithos;

import java.util.Arrays;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * Handles isometric transformations (Rademacher preconditioning + block
 * Walsh-Hadamard rotation)
 * and binarization.
 */
public class TransformOperator {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private final int dimension;
    private final int[] tiers;
    private final float[] signs;

    public TransformOperator(int dimension, int[] tiers) {
        this.dimension = dimension;
        this.tiers = tiers;
        this.signs = new float[dimension];

        // Generate Rademacher signs deterministically using seed 42 (matching Python
        // side)
        Random rand = new Random(42);
        for (int i = 0; i < dimension; i++) {
            signs[i] = rand.nextBoolean() ? 1.0f : -1.0f;
        }
    }

    public TransformOperator(int dimension, int[] tiers, float[] customSigns) {
        this.dimension = dimension;
        this.tiers = tiers;
        if (customSigns == null || customSigns.length != dimension) {
            throw new IllegalArgumentException("Signs length must match dimension");
        }
        this.signs = customSigns;
    }

    /**
     * Compute the cumulative energy distribution from the LoRA weights.
     */
    public static float[] computeCumulativeEnergy(float[] flatW, int D, int D0) {
        // Construct symmetric covariance matrix A = W * W^T
        float[][] A = new float[D][D];
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                float sum = 0.0f;
                int k = 0;
                int kBound = SPECIES.loopBound(D0);
                FloatVector vSum = FloatVector.zero(SPECIES);
                for (; k < kBound; k += SPECIES.length()) {
                    FloatVector va = FloatVector.fromArray(SPECIES, flatW, i * D0 + k);
                    FloatVector vb = FloatVector.fromArray(SPECIES, flatW, j * D0 + k);
                    vSum = va.fma(vb, vSum);
                }
                sum = vSum.reduceLanes(VectorOperators.ADD);
                for (; k < D0; k++) {
                    sum += flatW[i * D0 + k] * flatW[j * D0 + k];
                }
                A[i][j] = sum;
            }
        }

        // Jacobi eigenvalue algorithm
        float[] eigenvalues = jacobiEigenvalues(A, D);

        // Singular values are square roots of eigenvalues
        float[] sigmas = new float[D];
        float sumSigmasSq = 0.0f;
        for (int i = 0; i < D; i++) {
            sigmas[i] = (float) Math.sqrt(Math.max(0.0, eigenvalues[i]));
            sumSigmasSq += sigmas[i] * sigmas[i];
        }

        // Sort sigmas in descending order
        Arrays.sort(sigmas);
        for (int i = 0; i < D / 2; i++) {
            float temp = sigmas[i];
            sigmas[i] = sigmas[D - 1 - i];
            sigmas[D - 1 - i] = temp;
        }

        // Compute cumulative spectral energy
        float[] phi = new float[D];
        float runningSum = 0.0f;
        for (int i = 0; i < D; i++) {
            runningSum += sigmas[i] * sigmas[i];
            phi[i] = sumSigmasSq > 0 ? (runningSum / sumSigmasSq) : 0.0f;
        }
        return phi;
    }

    private static float[] jacobiEigenvalues(float[][] A, int n) {
        float[] d = new float[n];
        for (int i = 0; i < n; i++)
            d[i] = A[i][i];

        int maxIterations = 100;
        for (int iter = 0; iter < maxIterations; iter++) {
            int p = 0, q = 1;
            float maxVal = Math.abs(A[0][1]);
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (Math.abs(A[i][j]) > maxVal) {
                        maxVal = Math.abs(A[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxVal < 1e-6f)
                break;

            float apq = A[p][q];
            float app = A[p][p];
            float aqq = A[q][q];

            float theta = 0.5f * (aqq - app) / apq;
            float t = (float) (1.0 / (Math.abs(theta) + Math.sqrt(1.0 + theta * theta)));
            if (theta < 0)
                t = -t;

            float c = (float) (1.0 / Math.sqrt(1.0 + t * t));
            float s = t * c;

            A[p][q] = 0.0f;
            A[p][p] = app - t * apq;
            A[q][q] = aqq + t * apq;

            for (int i = 0; i < n; i++) {
                if (i != p && i != q) {
                    float aip = A[i][p];
                    float aiq = A[i][q];
                    A[i][p] = c * aip - s * aiq;
                    A[p][i] = A[i][p];
                    A[i][q] = s * aip + c * aiq;
                    A[q][i] = A[i][q];
                }
            }
        }

        float[] eigenvalues = new float[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = A[i][i];
        }
        return eigenvalues;
    }

    /**
     * Preconditions a vector with Rademacher signs, rotates via block-diagonal
     * Hadamard,
     * and binarizes it into a packed long array.
     */
    public long[] transformAndQuantize(float[] x) {
        if (x.length != dimension) {
            throw new IllegalArgumentException("Input vector size " + x.length + " must match dimension " + dimension);
        }

        // 1. Rademacher Preconditioning (Sign-flip)
        float[] z = new float[dimension];
        int i = 0;
        int upperBound = SPECIES.loopBound(dimension);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, x, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, signs, i);
            va.mul(vb).intoArray(z, i);
        }
        for (; i < dimension; i++) {
            z[i] = x[i] * signs[i];
        }

        // 2. Block-Diagonal Hadamard Rotation
        int start = 0;
        for (int tier : tiers) {
            int width = tier - start;
            rotateBlock(z, start, width);
            start = tier;
        }

        // 3. 1-Bit Binarization & Packing into longs (64 bits per long)
        int numLongs = (dimension + 63) / 64;
        long[] packed = new long[numLongs];
        for (int j = 0; j < dimension; j++) {
            if (z[j] >= 0.0f) {
                int longIdx = j / 64;
                int bitIdx = j % 64;
                packed[longIdx] |= (1L << bitIdx);
            }
        }
        return packed;
    }

    /**
     * Back-projects a target transformed vector z to raw input space x.
     * Since H_BD and D_pre are orthogonal/symmetric, back-projecting is
     * self-inverse.
     */
    public float[] backProject(float[] z) {
        if (z.length != dimension) {
            throw new IllegalArgumentException("Target vector size must match dimension");
        }
        float[] x = new float[dimension];
        System.arraycopy(z, 0, x, 0, dimension);

        // 1. Rotate by block Hadamard
        int start = 0;
        for (int tier : tiers) {
            int width = tier - start;
            rotateBlock(x, start, width);
            start = tier;
        }

        // 2. Precondition (sign-flip)
        int idx = 0;
        int upper = SPECIES.loopBound(dimension);
        for (; idx < upper; idx += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, x, idx);
            FloatVector vb = FloatVector.fromArray(SPECIES, signs, idx);
            va.mul(vb).intoArray(x, idx);
        }
        for (; idx < dimension; idx++) {
            x[idx] = x[idx] * signs[idx];
        }
        return x;
    }

    private void rotateBlock(float[] z, int start, int width) {
        // Check if width is power of two
        if ((width & (width - 1)) == 0) {
            fwht(z, start, width);
        } else {
            // Factorize width = u * v where u is the largest power of two
            int u = 1;
            while (width % (u * 2) == 0) {
                u *= 2;
            }
            int v = width / u;
            kroneckerRotate(z, start, u, v);
        }
    }

    private void fwht(float[] a, int start, int length) {
        for (int len = 1; len < length; len <<= 1) {
            for (int i = 0; i < length; i += (len << 1)) {
                if (len >= SPECIES.length()) {
                    int j = 0;
                    int lenBound = SPECIES.loopBound(len);
                    for (; j < lenBound; j += SPECIES.length()) {
                        FloatVector vu = FloatVector.fromArray(SPECIES, a, start + i + j);
                        FloatVector vv = FloatVector.fromArray(SPECIES, a, start + i + len + j);
                        FloatVector vAdd = vu.add(vv);
                        FloatVector vSub = vu.sub(vv);
                        vAdd.intoArray(a, start + i + j);
                        vSub.intoArray(a, start + i + len + j);
                    }
                    for (; j < len; j++) {
                        float u = a[start + i + j];
                        float v = a[start + i + len + j];
                        a[start + i + j] = u + v;
                        a[start + i + len + j] = u - v;
                    }
                } else {
                    for (int j = 0; j < len; j++) {
                        float u = a[start + i + j];
                        float v = a[start + i + len + j];
                        a[start + i + j] = u + v;
                        a[start + i + len + j] = u - v;
                    }
                }
            }
        }
        // Orthogonal normalization
        float scale = (float) (1.0 / Math.sqrt(length));
        int idx = 0;
        int upper = SPECIES.loopBound(length);
        for (; idx < upper; idx += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, start + idx);
            va.mul(scale).intoArray(a, start + idx);
        }
        for (; idx < length; idx++) {
            a[start + idx] *= scale;
        }
    }

    private void kroneckerRotate(float[] a, int start, int u, int v) {
        // Construct deterministic orthogonal matrix Omega_v using DCT basis
        float[][] omega = new float[v][v];
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                if (i == 0) {
                    omega[i][j] = (float) (1.0 / Math.sqrt(v));
                } else {
                    omega[i][j] = (float) (Math.sqrt(2.0 / v) * Math.cos(Math.PI * i * (2 * j + 1) / (2.0 * v)));
                }
            }
        }

        // Apply Omega_v to each block of size v
        float[] temp = new float[u * v];
        for (int block = 0; block < u; block++) {
            int blockStart = start + block * v;
            for (int i = 0; i < v; i++) {
                float sum = 0.0f;
                for (int j = 0; j < v; j++) {
                    sum += omega[i][j] * a[blockStart + j];
                }
                temp[block * v + i] = sum;
            }
        }

        // Apply FWHT of size u to each coordinate across blocks
        float[] column = new float[u];
        for (int coord = 0; coord < v; coord++) {
            // Gather
            for (int block = 0; block < u; block++) {
                column[block] = temp[block * v + coord];
            }

            // Transform
            fwht(column, 0, u);

            // Scatter back to a
            for (int block = 0; block < u; block++) {
                a[start + block * v + coord] = column[block];
            }
        }
    }
}
