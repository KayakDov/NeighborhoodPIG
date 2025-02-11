package JCudaWrapper.array;

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

        if (ld() > 1) {
            DArray1d gpuArray = new DArray1d(dst.length);
            get(handle, gpuArray);
            handle.synch();
            gpuArray.get(handle, Pointer.to(dst));
        } else get(handle, Pointer.to(dst));

    }

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     */
    public default double[] get(Handle handle) {

        double[] export = new double[size()];

        if (ld() > 1) 
            try (DArray1d gpuArray = new DArray1d(size())) {
                get(handle, gpuArray);
                gpuArray.get(handle, Pointer.to(export));
            }
        else get(handle, Pointer.to(export));

        
        handle.synch();

        return export;
    }

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

    

    

    /**
     * Fills a matrix with a scalar value directly on the GPU using a CUDA
     * kernel.
     *
     * This function sets all elements of the matrix A to the given scalar
     * value. The matrix A is stored in column-major order, and the leading
     * dimension of A is specified by lda.
     *
     * In contrast to the method that doesn't use a handle, this one
     *
     * @param handle A handle.
     * @param fill the scalar value to set all elements of A
     * @param inc The increment with which the method iterates over the array.
     * @param lineLength The number of elements in each line.
     * @param ld The number of elements between the first element of each row.
     * @return this;
     */
    public default DArray fill(Handle handle, double fill, int inc, int lineLength, int ld) {
        Kernel.run("fill", handle, n(inc), this, P.to(inc), P.to(lineLength), P.to(ld), P.to(fill));
        return this;
    }


    /**
     * The maximum number of times something can be done at the requested
     * increment.
     *
     * @param inc The increment between the elements that the something is done
     * with.
     * @return The number of times is can be done
     */
    default int n(int inc) {
        return (int) Math.ceil((double) size() / inc);
    }

    /**
     * Breaks this array into a a set of sub arrays, one after the other.
     *
     * @param batchSize The number of elements in the batch.
     * @param strideSize
     * @param subArrayLength The number of elements in each subArray.
     * @return A representation of this array as a set of sub arrays.
     */
    public default DStrideArray getAsBatch(int subArrayLength, int strideSize, int batchSize) {
        return new DStrideArray1d(this, strideSize, subArrayLength, batchSize);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public default DSingleton get(int i) {
        return new DSingleton(this, i);
    }

    /**
     * Sets this array to the values in the source cpu array.
     *
     * @param handle The cntext.
     * @param srcCPUArray A cpu array. It should be the same length as this
     * array.
     * @return this array.
     */
    public default DArray set(Handle handle, double[] srcCPUArray) {
        set(handle, Pointer.to(srcCPUArray));
        return this;
    }
}
