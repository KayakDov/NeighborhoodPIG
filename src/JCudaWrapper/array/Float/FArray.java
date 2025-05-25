package JCudaWrapper.array.Float;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;

/**
 * Represents an array stored in GPU memory, providing methods for mathematical
 * operations and data transfer, specifically for float precision.
 *
 * @author E. Dov Neimand
 */
public interface FArray extends Array {

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param dst Where the array is to be copied to.
     */
    public default void get(Handle handle, float[] dst) {
        if (ld() > 1) {
            try (FArray1d gpuArray = new FArray1d(dst.length)) {
                get(handle, (FArray1d)gpuArray);
                handle.synch();
                gpuArray.get(handle, Pointer.to(dst));
            }
        } else
            get(handle, Pointer.to(dst));
    }

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     */
    public float[] get(Handle handle);

    /**
     * Copies to here with increments.
     *
     * @param handle The context.
     * @param src Where the data is copied from.
     * @return this.
     */
    public default FArray set(Handle handle, FArray src) {
        src.get(handle, this);
        return this;
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    public default FArray fill(Handle handle, float fill) {
        throw new UnsupportedOperationException("TODO: Need to write a fill kernel for floats.");
    }

    /**
     * Breaks this array into a a set of sub arrays, one after the other.
     *
     * @param batchSize The number of elements in the batch.
     * @param strideSize
     * @param subArrayLength The number of elements in each subArray.
     * @return A representation of this array as a set of sub arrays.
     */
    public default FStrideArray1d getAsBatch(int subArrayLength, int strideSize, int batchSize) {
        return new FStrideArray1d(this, strideSize, subArrayLength, batchSize);
    }

    /**
     * Sets this array to the values in the source CPU array.
     *
     * @param handle The context.
     * @param srcCPUArray A CPU array. It should be the same length as this
     * array.
     * @return this array.
     */
    public default FArray set(Handle handle, float... srcCPUArray) {
        set(handle, Pointer.to(srcCPUArray));
        return this;
    }

    /**
     * Performs matrix addition or subtraction.
     *
     * @param handle the cuBLAS context handle
     * @param transA specifies whether matrix A is transposed
     * @param transB specifies whether matrix B is transposed
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @return this
     */
    public default FArray setSum(Handle handle, boolean transA, boolean transB, float alpha, FArray a, float beta, FArray b) {
        opCheck(JCublas2.cublasSgeam(handle.get(),
                Array.transpose(transA), Array.transpose(transB),
                entriesPerLine(), linesPerLayer(),
                P.to(alpha), a.pointer(), a.ld(),
                P.to(beta), b == null ? new Pointer() : b.pointer(), b == null ? 1 : b.ld(),
                pointer(), ld()
        ));
        return this;
    }

    /**
     * A one-dimensional representation of this array.
     *
     * @return A 1D representation of this array.
     */
    public default FArray1d as1d() {
        return new FArray1d(this, 0, size(), ld());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public default FArray2d as2d() {
        return new FArray2d(this, entriesPerLine());
    }

    /**
     * A 3D representation of this array.
     *
     * @param linesPerLayer Number of lines per layer.
     * @return A 3D representation of this array.
     */
    public default FArray3d as3d(int linesPerLayer) {
        return new FArray3d(this, entriesPerLine(), linesPerLayer);
    }

//    /**
//     * Scales this vector by the scalar multiplier:
//     *
//     * <pre>
//     * this = mult * this
//     * </pre>
//     *
//     * @param handle handle to the cuBLAS library context.
//     * @param scalar Scalar multiplier applied to vector X.
//     * @param src Where the array is copied from. The result is placed here.
//     * @return this;
//     */
//    public FArray setProduct(Handle handle, float scalar, FArray src);
}
