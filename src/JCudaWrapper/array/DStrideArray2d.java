package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.cudaError;

/**
 *
 * @author E. Dov Neimand
 */
public class DStrideArray2d extends DArray2d implements StrideArray {

    /**
     * Creates a bunch of consecutive 2d arrays.
     * @param numLines The number of lines in each array.
     * @param entriesPerLine The number of entries in each line.
     * @param batchSize The number of 2d arrays.
     */
    public DStrideArray2d(int entriesPerLine, int numLines, int batchSize) {
        super(entriesPerLine, numLines*batchSize);
        this.stride = linesPerLayer()*ld();
        this.batchSize = batchSize;
        this.subArraySize = linesPerLayer() * entriesPerLine();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public final int linesPerLayer() {
        return super.linesPerLayer()/batchSize;
    }
    
    

    
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
    public void addProduct(Handle handle, boolean transA, boolean transB, double timesAB, DStrideArray2d matA, DStrideArray2d matB, double timesResult) {

        int result = JCublas2.cublasDgemmStridedBatched(handle.get(),
                Array.transpose(transA), Array.transpose(transB),
                matA.entriesPerLine(), matB.linesPerLayer(), matA.linesPerLayer(),
                P.to(timesAB),
                matA.pointer(), matA.ld(), matA.strideLines(),
                matB.pointer(), matB.ld(), matB.strideLines(),
                P.to(timesResult), pointer(), ld(), strideLines(),
                batchSize()
        );
        if (result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda multiplication failed.");

    }
    
    public final int stride, batchSize, subArraySize;

    /**
     * {@inheritDoc }
     */
    @Override
    public int size() {
        return super.size()*batchSize();
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d as1d() {
        return new DArray1d(this, 0, size(), 1);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray2d as2d() {
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray3d as3d(int linesPerLayer) {
        return new DArray3d(this, entriesPerLine(), linesPerLayer);
    }

    public DStrideArray2d set(Handle handle, DStrideArray2d from) {
        super.set(handle, from); 
        return this;
    }
    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray2d copy(Handle handle) {
        return new DStrideArray2d(entriesPerLine(), linesPerLayer(), batchSize())
                .set(handle, this);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int strideLines() {
        return stride;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int batchSize() {
        return batchSize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int subArraySize() {
        return subArraySize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray2d getSubArray(int arrayIndex) {
        return new DArray2d(this, 0, entriesPerLine(), arrayIndex*ld()*linesPerLayer(), linesPerLayer());
    }

}
