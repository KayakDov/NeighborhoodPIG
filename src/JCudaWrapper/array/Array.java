package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaPitchedPtr;

/**
 *
 * @author dov
 */
public interface Array extends AutoCloseable {

    /**
     * The pointer to this array.
     *
     * @return The pointer to this array.
     */
    Pointer pointer();

    /**
     * The pitched pointer for this array.
     *
     * @return The pitched pointer for this array.
     */
    cudaPitchedPtr pitchedPointer();

    /**
     * The number of bytes consumed by each element in this array.
     *
     * @return The number of bytes consumed by each element in this array.
     */
    int bytesPerElement();

    /**
     * Returns a pointer to the element at the specified index in this array.
     *
     * @param offset The index of the element.
     * @return A CUdeviceptr pointing to the specified index.
     *
     * @throws ArrayIndexOutOfBoundsException if the index is out of bounds.
     */
    default Pointer pointer(int offset) {
        return pointer().withByteOffset(offset * bytesPerElement());
    }

    /**
     * removes this array's ID from the pool of stored array IDs. If this
     * array's ID is not stored then a runtime exception is thrown.
     */
    public void removeID();

    /**
     * Frees the GPU memory allocated for this array if it has not already been
     * freed.
     */
    @Override
    public default void close() {

        JCuda.cudaFree(pointer());
        removeID();
    }

    public int numELements();

    /**
     * Sets the contents of this array to 0.
     *
     * @param handle The handle.
     * @return this.
     */
    public default Array fill0(Handle handle) {
        int error = JCuda.cudaMemsetAsync(pointer(), 0, numELements() * bytesPerElement(), handle.getStream());
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        return this;
    }

    /**
     * Copies data from this GPU array to another GPU array. We assume the
     * target array is the same size as this array. If not, a sub array should
     * be created from the larger of the two arrays.
     *
     * @param to The destination GPU array.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void get(Handle handle, Array to);
    
    /**
     * Copies data from this GPU array to another GPU array. We assume the
     * target array is the same size as this array. If not, a sub array should
     * be created from the larger of the two arrays.
     *
     * If you do not wish to copy the entire array, then create a sub array and
     * call get on that sub array.
     * 
     * @param cpuPointer The destination CPU array.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    void get(Handle handle, Pointer cpuPointer);


    /**
     * Copies data to this GPU array from another GPU array. Allows for multiple
     * copying in parallel.
     *
     * @param from The source GPU array.
     
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public default void set(Handle handle, Array from){
        from.get(handle, this);
    }

    /**
     * Copies data to this GPU array from another GPU array. Allows for multiple
     * copying in parallel.
     *
     * If you wish to copy the array into a specific destination, then use a sub array of this one.
     * 
     * @param fromCPU The source CPU array.     
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void set(Handle handle, Pointer fromCPU);

}
