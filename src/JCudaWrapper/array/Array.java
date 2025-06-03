package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import jcuda.Pointer;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

/**
 *
 * @author dov
 */
public interface Array extends AutoCloseable {

    static Set<Pointer> allocatedArrays = Collections.synchronizedSet(new HashSet<>());

    /**
     * This is a debugging tool. Records that memory has been allocated. This
     * should be called every time memory is allocated and is a tool to prevent
     * memory leaks.
     *
     * @param p The pointer to the allocated memory.
     */
    public static void recordMemAloc(Pointer p) {
        allocatedArrays.add(p);
//        System.out.println("JCudaWrapper.array.Array.recordMemAloc() " + p.toString());
//        (new RuntimeException("Allocate " + p.toString() + "TODO: delete me")).printStackTrace(System.out);
    }

    /**
     * Fill modes. Use lower to indicate a lower triangle, upper to indicate an
     * upper triangle, and full for full triangles.
     */
    public static final int LOWER = cublasFillMode.CUBLAS_FILL_MODE_LOWER,
            UPPER = cublasFillMode.CUBLAS_FILL_MODE_UPPER,
            FULL = cublasFillMode.CUBLAS_FILL_MODE_FULL;

    /**
     * The pointer to this array.
     *
     * @return The pointer to this array.
     */
    Pointer pointer();

    /**
     * The number of bytes consumed by each element in this array.
     *
     * @return The number of bytes consumed by each element in this array.
     */
    int bytesPerEntry();

    /**
     * Returns a pointer to the element at the specified index in this array.
     *
     * @param offset The index of the element.
     * @return A Pointer pointing to the specified index.
     *
     * @throws ArrayIndexOutOfBoundsException if the index is out of bounds.
     */
    default Pointer pointer(int offset) {

        return pointer().withByteOffset(offset * bytesPerEntry());
    }

    /**
     * Frees the GPU memory allocated for this array if it has not already been
     * freed.
     */
    @Override
    public default void close() {
//        (new RuntimeException("Close " + pointer().toString() + "TODO: delete me")).printStackTrace(System.out);
        Pointer ptr = pointer();
        if (!allocatedArrays.remove(ptr))
            throw new RuntimeException("Trying to remove a pointer, (" + ptr.toString() + ") that does not exist or has already been removed.");
        JCuda.cudaFree(ptr);
        
    }

    /**
     * The number of elements, not bytes, stored here.
     *
     * @return The number of elements, not bytes, stored here.
     */
    public int size();

    /**
     * Sets every element of this array to 0.
     *
     * @param handle The handle.
     * @return this.
     */
    public default Array fill0(Handle handle) {
        opCheck(JCuda.cudaMemsetAsync(pointer(), 0, size() * bytesPerEntry(), handle.getStream()));
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
     */
    void get(Handle handle, Pointer cpuPointer);

    /**
     * Copies data to this GPU array from another GPU array. Allows for multiple
     * copying in parallel.
     *
     * @param from The source GPU array.
     *
     * @param handle The handle.
     * @return this
     *
     */
    public default Array set(Handle handle, Array from) {
        from.get(handle, this);
        return this;
    }

    /**
     * Copies data to this GPU array from another GPU array. Allows for multiple
     * copying in parallel.
     *
     * If you wish to copy the array into a specific destination, then use a sub
     * array of this one.
     *
     * @param fromCPU The source CPU array.
     * @param handle The handle.
     * @return this
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public Array set(Handle handle, Pointer fromCPU);

    /**
     * The element at the index. If this array is 3d, then it's effectively
     * flattened for this purpose.
     *
     * Note, no new memory is allocated. This singleton is a pointer to existing
     * memory.
     *
     * @param index The index of the desired element. Padding is resolved
     * internally and should not taken into account when choosing a value for
     * index.
     * @return A singleton of the element at the desired index.
     */
    public Singleton get(int index);

    /**
     * Returns the appropriate cuBLAS operation flag for transposition.
     *
     * @param isTranspose Whether the operation should transpose the matrix.
     * @return `CUBLAS_OP_T` if true, `CUBLAS_OP_N` otherwise.
     */
    public static int transpose(boolean isTranspose) {
        return isTranspose ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
    }

    /**
     * If the operation was not a success, then an exception is thrown.
     *
     * @param errorCode The result of the cuda operation.
     */
    default void opCheck(int errorCode) {
        if (errorCode != cudaError.cudaSuccess)
            throw new RuntimeException("Operation failed: " + cudaError.stringFor(errorCode));
    }

    /**
     * The number of lines in the array. It's 1 for 1d arrays, and more for 2d
     * or 3d arrays.
     *
     * @return 1
     */
    public default int linesPerLayer() {
        return size();
    }

    /**
     * The number of lines in the array. It's 1 for 1d arrays, and more for 2d
     * or 3d arrays.
     *
     * @return 1
     */
    public default int layersPerGrid() {
        return 1;
    }

    /**
     * The number of entries per line.
     *
     * @return The number of entries per line.
     */
    public default int entriesPerLine() {
        return 1;
    }

    /**
     * True if this matrix has one entry per line, one layer per grid, and any
     * number of lines per layer.
     *
     * @return True if this matrix has one entry per line, one layer per grid,
     * and any number of lines per layer.
     */
    public default boolean is1D() {
        return entriesPerLine() == 1 && layersPerGrid() == 1;
    }

    /**
     * If any of the conditions are not true, an exception is thrown.
     *
     * @param check These all must be true.
     */
    public default void confirm(boolean... check) {
        for (int i = 0; i < check.length; i++)
            if (!check[i])
                throw new IllegalArgumentException("Argument number " + i);
    }

    /**
     * A copy of this array that occupies its own memory.
     *
     * @param handle Used to copy the data.
     * @return A new array that is copied from this one.
     */
    public Array copy(Handle handle);

    /**
     * The number of elements that could fit in the pitch.
     *
     * @return The number of elements that could fit in the pitch.
     */
    public int ld();

    /**
     * Takes in the index of the desired element, and returns the actual index
     * of that element accounting for pitch.
     *
     * @param entryNumber The index of the desired element, without accounting
     * for pitch.
     * @return The index with accounting for pitch.
     */
    public default int memIndex(int entryNumber) {
        return (entryNumber / entriesPerLine()) * ld() + entryNumber % entriesPerLine();
    }

    /**
     * true if there is padding, ie ld() != entriesPerLine, and false otherwise.
     *
     * @return true if there is padding, ie ld() != entriesPerLine, and false
     * otherwise.
     */
    public default boolean hasPadding() {
        return ld() != 1 && (ld() <= entriesPerLine() || linesPerLayer() > 1);//ld() != entriesPerLine() && size() != 1;
    }

    /**
     * A one dimensional representation of this array.
     *
     * @return
     */
    public Array1d as1d();

    /**
     * A 2 dimensional representation of this array. If this array is already
     * 2d, then this array is returned. If it is 3d then each layer precedes the
     * previous layers.
     *
     * @return A 2 dimensional representation of this array.
     */
    public Array2d as2d();

    /**
     * A 3d representation of this array.
     *
     * @param linesPerLayer
     * @return A 3d representation of this array.
     */
    public Array3d as3d(int linesPerLayer);

    /**
     * The total memory used by this array.
     * @return The total memory used by this array.
     */
    public default long totalMemoryUsed(){
        return (size()/entriesPerLine())*ld()*bytesPerEntry();
    }
}
