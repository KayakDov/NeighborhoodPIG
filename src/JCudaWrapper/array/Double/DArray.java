package JCudaWrapper.array.Double;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Int.ISingleton;
import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;

/**
 * Represents an array stored in GPU memory, providing methods for mathematical
 * operations and data transfer.
 *
 * @author E. Dov Neimand
 */
public interface DArray extends Array {

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param dst Where the array is to be copied to.
     */
    public default void get(Handle handle, double[] dst) {

        if (hasPadding()) {
            try (DArray1d gpuArray = new DArray1d(dst.length)) {
                get(handle, gpuArray);
                handle.synch();
                gpuArray.get(handle, Pointer.to(dst));
            }
        } else get(handle, Pointer.to(dst));

    }

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     */
    public double[] get(Handle handle);

    /**
     * Copies to here with increments.
     *
     * @param handle The context.
     * @param src Where the data is copied from.
     * @return this.
     */
    public default DArray set(Handle handle, DArray src) {
        src.get(handle, this);
        return this;
    }

//    /**
//     * Fills a matrix with a scalar value directly on the GPU using a CUDA
//     * kernel.
//     *
//     * This function sets all elements of the matrix A to the given scalar
//     * value. The matrix A is stored in column-major order, and the leading
//     * dimension of A is specified by lda.
//     *
//     * In contrast to the method that doesn't use a handle, this one
//     *
//     * @param handle A handle.
//     * @param fill the scalar value to set all elements of A
//     * @return this;
//     */
//    public default DArray fill(Handle handle, double fill) {
//        Kernel.run("fill", handle, size(), this, P.to(ld()), P.to(entriesPerLine()), P.to(ld()), P.to(fill));
//        return this;
//    }
    /**
     * Breaks this array into a a set of sub arrays, one after the other.
     *
     * @param batchSize The number of elements in the batch.
     * @param strideSize
     * @param subArrayLength The number of elements in each subArray.
     * @return A representation of this array as a set of sub arrays.
     */
    public default DStrideArray1d getAsBatch(int subArrayLength, int strideSize, int batchSize) {
        return new DStrideArray1d(this, strideSize, subArrayLength, batchSize);
    }

    /**
     * Sets this array to the values in the source cpu array.
     *
     * @param handle The cntext.
     * @param srcCPUArray A cpu array. It should be the same length as this
     * array.
     * @return this array.
     */
    public default DArray set(Handle handle, double... srcCPUArray) {
        set(handle, Pointer.to(srcCPUArray));
        return this;
    }

    /**
     * Performs matrix addition or subtraction. a and b may be vectors (matrices
     * with 1 element per line) or matrices with identical number of elements
     * per line.
     *
     * <p>
     * This function computes C = alpha * A + beta * B, where A, B, and C are
     * matrices.
     * </p>
     *
     * @param handle the cuBLAS context handle
     * @param transA specifies whether matrix A is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param transB specifies whether matrix B is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @return this
     *
     */
    public default DArray setSum(Handle handle, boolean transA, boolean transB, double alpha, DArray a, double beta, DArray b) {

        opCheck(JCublas2.cublasDgeam(handle.get(),
                Array.transpose(transA), Array.transpose(transB),
                entriesPerLine(), linesPerLayer(),
                P.to(alpha), a.pointer(), a.ld(),
                P.to(beta), b == null ? new Pointer() : b.pointer(), b == null ? 1 : b.ld(),
                pointer(), ld()
        ));
        return this;
    }

    /**
     * A one dimensional representation of this array.
     *
     * @return
     */
    public default DArray1d as1d() {
        return new DArray1d(this, 0, size(), ld());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default DArray2d as2d() {
        return new DArray2d(this, entriesPerLine());
    }

    /**
     * A 3d representation of this array.
     *
     * @param linesPerLayer
     * @return A 3d representation of this array.
     */
    public default DArray3d as3d(int linesPerLayer) {
        return new DArray3d(this, entriesPerLine(), linesPerLayer);
    }

//    /**
//     * Scales this vector by the scalar mult:
//     *
//     * <pre>
//     * this = mult * this
//     * </pre>
//     *
//     * @param handle handle to the cuBLAS library context.
//     * @param scalar Scalar multiplier applied to vector X.
//     * @param src Where the array is copied from.  The result is placed here.
//     * @return this;
//     *
//     */
//    public DArray setProduct(Handle handle, double scalar, DArray src);
    /**
     * Gets the singleton at the desired index.
     *
     * @param index The index of the desired singleton.
     * @return The singleton at the desired index.
     */
    @Override
    public default DSingleton get(int index) {
        return new DSingleton(this, index);
    }

}
