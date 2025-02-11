package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.cudaError;

/**
 *
 * @author E. Dov Neimand
 */
public class DStrideArray2d extends Array2d implements DStrideArray {

    
    /**
     * Performs batched matrix-matrix multiplication:
     *
     * <pre>
     * Result[i] = alpha * op(A[i]) * op(B[i]) + timesResult * Result[i]
     * </pre>
     *
     * Where op(A) and op(B) can be A and B or their transposes.
     *
     * This method computes multiple matrix-matrix multiplications at once,
     * using strided data access, allowing for efficient batch processing.
     *
     * @param handle Handle to the cuBLAS library context.
     * @param transA True if matrix A should be transposed, false otherwise.
     * @param transB True if matrix B should be transposed, false otherwise.
     * @param timesAB Scalar multiplier applied to the matrix-matrix product.
     * @param matA Pointer to the batched matrix A in GPU memory.
     * @param matB Pointer to the batched matrix B in GPU memory.
     * @param timesResult Scalar multiplier applied to each result matrix before
     * adding the matrix-matrix product.
     *
     */
    public void addProduct(Handle handle, boolean transA, boolean transB, double timesAB, DStrideArray matA, DStrideArray matB, double timesResult) {

        int result = JCublas2.cublasDgemmStridedBatched(handle.get(),
                Array.transpose(transA), Array.transpose(transB),
                matA.entriesPerLine(), matB.linesPerLayer(), matA.linesPerLayer(),
                P.to(timesAB),
                matA.pointer(), matA.ld(), matA.stride(),
                matB.pointer(), matB.ld(), matB.stride(),
                P.to(timesResult), pointer(), ld(), stride(),
                batchSize()
        );
        if (result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda multiplication failed.");

    }
    
    public final int stride, batchSize, subArraySize;

    @Override
    public int size() {
        return super.size()*batchSize();
    }
    
    @Override
    public DArray1d as1d() {
        return new DArray1d(this, 0, size(), 1);
    }

    @Override
    public Array2d as2d() {
        return this;
    }

    @Override
    public DArray3d as3d(int linesPerLayer) {
        return new DArray3d(this, entriesPerLine(), linesPerLayer);
    }

    @Override
    public Array copy(Handle handle) {
        return new DStrideArray2d().set(handle, this);
    }

    @Override
    public int stride() {
        return stride;
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    @Override
    public int subArraySize() {
        return subArraySize;
    }

    @Override
    public DArray2d getSubArray(int arrayIndex) {
        return new DArray2d(this, arrayIndex*ld()*linesPerLayer(), linesPerLayer(), 0, entriesPerLine());
    }

}
