package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.ResourceDealocator;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import JCudaWrapper.resourceManagement.Handle;

/**
 * Abstract class representing a GPU array that stores data on the GPU. This
 * class provides various utilities for managing GPU memory and copying data
 * between the host (CPU) and device (GPU).
 *
 * <p>
 * It supports different primitive types including byte, char, double, float,
 * int, long, and short. The derived classes should implement specific
 * functionalities for each type of data.</p>
 *
 * <p>
 * Note: This class is designed to be extended by specific array
 * implementations.</p>
 *
 * http://www.jcuda.org/tutorial/TutorialIndex.html#CreatingKernels
 *
 * TODO:Implement methods with JBLAS as an aulternative for when there's no gpu.
 *
 * @author E. Dov Neimand
 */
abstract class Array implements AutoCloseable {

    private final Cleaner.Cleanable cleanable;
    /**
     * The pointer to the array in gpu memory.
     */
    protected final CUdeviceptr pointer;
    /**
     * The length of the array.
     */
    public final int length;
    private final PrimitiveType type;

    /**
     * Enum representing different primitive types and their sizes.
     */
    public enum PrimitiveType {
        BYTE(Sizeof.BYTE), CHAR(Sizeof.CHAR), DOUBLE(Sizeof.DOUBLE), FLOAT(Sizeof.FLOAT),
        INT(Sizeof.INT), LONG(Sizeof.LONG), SHORT(Sizeof.SHORT), POINTER(Sizeof.POINTER);

        public final int size;

        /**
         * Constructs a PrimitiveType enum.
         *
         * @param numBytes The number of bytes per element.
         */
        private PrimitiveType(int numBytes) {
            this.size = numBytes;
        }
    }

    /**
     * Constructs an Array with the given GPU pointer, length, and element type.
     *
     * @param p The pointer to the GPU memory allocated for this array.
     * @param length The length of the array.
     * @param type The type of elements in the array.
     *
     * @throws IllegalArgumentException if the pointer is null or length is
     * negative.
     */
    protected Array(CUdeviceptr p, int length, PrimitiveType type) {
        checkNull(p, type);
        checkPositive(length);

        this.pointer = p;
        this.length = length;
        this.type = type;

        // Register cleanup of GPU memory
        cleanable = ResourceDealocator.register(this, pointer -> JCuda.cudaFree(pointer), pointer);
    }

    /**
     * Creates a copy of this array.
     *
     * @param handle The handle.
     * @return A new Array instance with the same data.
     */
    public abstract Array copy(Handle handle);

    /**
     * Returns a pointer to the element at the specified index in this array.
     *
     * @param index The index of the element.
     * @return A CUdeviceptr pointing to the specified index.
     *
     * @throws ArrayIndexOutOfBoundsException if the index is out of bounds.
     */
    protected CUdeviceptr pointer(int index) {
        checkPositive(index);
        checkAgainstLength(index);

        return pointer.withByteOffset(index * type.size);
    }

    /**
     * Allocates GPU memory for an array of the specified size and type.
     *
     * @param numElements The number of elements to allocate space for.
     * @param type The type of the array elements.
     * @return A CUdeviceptr pointing to the allocated memory.
     *
     * @throws IllegalArgumentException if numElements is negative.
     */
    protected static CUdeviceptr empty(int numElements, PrimitiveType type) {
        checkPositive(numElements);

        CUdeviceptr p = new CUdeviceptr();
        int error = JCuda.cudaMalloc(p, numElements * type.size);
        if(error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        
        return p;
    }

    /**
     * Copies data from a CPU array to a GPU array.
     *
     * @param to The destination GPU array.
     * @param fromCPUArray The source CPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param length The number of elements to copy.
     * @param type The type of the elements.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    protected static void copy(Handle handle, Array to, Pointer fromCPUArray, int toIndex, int fromIndex, int length, PrimitiveType type) {
        checkPositive(toIndex, fromIndex, length);
        to.checkAgainstLength(toIndex + length - 1);

        int result = JCuda.cudaMemcpyAsync(
                to.pointer.withByteOffset(toIndex * type.size),
                fromCPUArray.withByteOffset(fromIndex * type.size),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                handle.getStream()
        );

        if (result != cudaError.cudaSuccess) {
            throw new RuntimeException("CUDA error: " + JCuda.cudaGetErrorString(result));
        }

    }

    /**
     * Copies data from this GPU array to another GPU array.
     *
     * @param to The destination GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void get(Handle handle, Array to, int toIndex, int fromIndex, int length) {
        checkPositive(toIndex, fromIndex, length);
        to.checkAgainstLength(toIndex + length - 1);
        checkAgainstLength(fromIndex + length - 1);
        checkNull(to);

        int error = JCuda.cudaMemcpyAsync(to.pointer(toIndex),
                pointer(fromIndex),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                handle.getStream()
        );
        if(error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * Copies data from this GPU array to another GPU array. Allows for multiple
     * copying in parallel.
     *
     * @param to The destination GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void get(Array to, int toIndex, int fromIndex, int length, Handle handle) {
        int error = JCuda.cudaMemcpyAsync(
                to.pointer(toIndex),
                pointer(fromIndex),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                handle.getStream());
        if(error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * Copies data to this GPU array from another GPU array. Allows for multiple
     * copying in parallel.
     *
     * @param from The source GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void set(Handle handle, Array from, int toIndex, int fromIndex, int length) {
        from.get(this, toIndex, fromIndex, length, handle);
    }

    /**
     * Copies data from this GPU array to a CPU array.
     *
     * @param toCPUArray The destination CPU array.
     * @param toStart The starting index in the CPU array.
     * @param fromStart The starting index in this GPU array.
     * @param length The number of elements to copy.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    protected void get(Handle handle, Pointer toCPUArray, int toStart, int fromStart, int length) {
        checkPositive(toStart, fromStart, length);
        checkAgainstLength(fromStart + length - 1);
        //TODO:  cudaHostAlloc can be faster, but has risks.        
        int error = JCuda.cudaMemcpyAsync(
                toCPUArray.withByteOffset(toStart * type.size),
                pointer(fromStart),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                handle.getStream()
        );
        if(error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * Copies data from a CPU array to this GPU array.
     *
     * @param fromCPUArray The source CPU array.
     * @param toIndex The starting index in this GPU array.
     * @param fromIndex The starting index in the CPU array.
     * @param size The number of elements to copy.
     * @param handle The handle.
     * @throws ArrayIndexOutOfBoundsException if any index is out of bounds or
     * size is negative.
     */
    protected void set(Handle handle, Pointer fromCPUArray, int toIndex, int fromIndex, int size) {
        copy(handle, this, fromCPUArray, toIndex, fromIndex, size, type);
    }

    /**
     * Copies data from a CPU array to this GPU array starting from the
     * beginning.
     *
     * @param fromCPUArray The source CPU array.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws ArrayIndexOutOfBoundsException if length is negative.
     */
    protected void set(Handle handle, Pointer fromCPUArray, int length) {

        set(handle, fromCPUArray, 0, 0, length);
    }

    /**
     * Copies data from a CPU array to this GPU array with a specified starting
     * index.
     *
     * @param fromCPUArray The source CPU array.
     * @param toIndex The starting index in this GPU array.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    protected void set(Handle handle, Pointer fromCPUArray, int toIndex, int length) {
        checkPositive(toIndex, length);
        checkAgainstLength(toIndex + length);

        set(handle, fromCPUArray, toIndex, 0, length);
    }

    /**
     * Frees the GPU memory allocated for this array. This method is invoked
     * automatically when the object is closed.
     */
    @Override
    public void close() {
        cleanable.clean();
    }

    /**
     * Sets the contents of this array to 0.
     *
     * @param handle The handle.
     * @return this.
     */
    public Array fill0(Handle handle) {
        int error = JCuda.cudaMemsetAsync(pointer, 0, length * type.size, handle.getStream());
        if(error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        return this;
    }

    /**
     * Checks that all the numbers are positive. An exception is thrown if not.
     *
     * @param maybePos A number that might be positive.
     */
    protected static void checkPositive(int... maybePos) {
        checkLowerBound(0, maybePos);
    }

    /**
     * Checks that all the numbers are greater than or equal to the lower bound.
     * An exception is thrown if not.
     *
     * @param bound The lower bound.
     * @param needsCheck A number that might be greater than the lower bound.
     */
    protected static void checkLowerBound(int bound, int... needsCheck) {
        if (Arrays.stream(needsCheck).anyMatch(l -> bound > l))
            throw new ArrayIndexOutOfBoundsException();
    }

    /**
     * Checks if any of the indeces are out of bounds, that is, greater than or
     * equal to the length.
     *
     * @param maybeInBounds A number that might or might not be out of bounds.
     */
    protected void checkAgainstLength(int... maybeInBounds) {
        checkUpperBound(length, maybeInBounds);
    }

    /**
     * Checks if any of the indeces are out of bounds, that is, greater than or
     * equal to the upper bound
     *
     * @param bound ALl elements passed must be less than this element..
     * @param maybeInBounds A number that might or might not be out of bounds.
     */
    protected void checkUpperBound(int bound, int... maybeInBounds) {
        for (int needsChecking : maybeInBounds)
            if (needsChecking >= bound)
                throw new ArrayIndexOutOfBoundsException(needsChecking + " is greater than the bound " + bound);
    }

    /**
     * Checks if any of the objects are null. If they are a NullPointerException
     * is thrown.
     *
     * @param maybeNull Objects that might be null.
     */
    protected static void checkNull(Object... maybeNull) {
        if (Arrays.stream(maybeNull).anyMatch(l -> l == null))
            throw new NullPointerException();
    }


    /**
     * A mapping from boolean transpose to the corresponding cuda integer.
     *
     * @param t True for transpose and false to not transpose.
     * @return An integer representing yes or no on a transpose operation.
     */
    static int transpose(boolean t) {
        return t ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
    }

    
    /**
     * A pointer to this pointer.
     * @return A pointer to this pointer.
     */
    public Pointer pointerToPointer(){
        return Pointer.to(pointer);
    }
}