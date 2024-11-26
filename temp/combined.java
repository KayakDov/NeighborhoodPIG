------------------------------------------------
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
     * @param n The number of elements to copy.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    protected void get(Handle handle, Pointer toCPUArray, int toStart, int fromStart, int n) {
        checkPositive(toStart, fromStart, n);
        checkAgainstLength(fromStart + n - 1);
       
        int error = JCuda.cudaMemcpyAsync(
                toCPUArray.withByteOffset(toStart * type.size),
                pointer(fromStart),
                n * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                handle.getStream()
        );
        if(error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + error + " " + cudaError.stringFor(error));
    }//TODO:  cudaHostAlloc can be faster, but has risks.

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
     * Default of true, and set to false when the memory is dealocated.
     */
    protected boolean isOpen = true;
    
    /**
     * Frees the GPU memory allocated for this array. This method is invoked
     * automatically when the object is closed.
     */
    @Override
    public void close() {
        cleanable.clean();
        isOpen = false;
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
     * Checks if the memory is available.  If it is not, then an exception is thrown.
     */
    public void checkMemAllocation(){
        if(!isOpen) throw new RuntimeException("This memory has been dealocated.");
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
}------------------------------------------------
package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import java.awt.image.Raster;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasDiagType;
import jcuda.jcublas.cublasFillMode;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import org.apache.commons.math3.exception.DimensionMismatchException;

/**
 * This class provides functionalities to create and manipulate double arrays on
 * the GPU.
 *
 * For more methods that might be useful here, see:
 * https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
 *
 * TODO: create arrays other than double.
 *
 * @author E. Dov Neimand
 */
public class DArray extends Array {

    /**
     * Creates a GPU array from a CPU array.
     *
     * @param handle The gpu handle that manages this operation. The handle is
     * not saved by the class and should be synched and closed externally.
     * @param values The array to be copied to the GPU.
     * @throws IllegalArgumentException if the values array is null.
     */
    public DArray(Handle handle, double... values) {
        this(Array.empty(values.length, PrimitiveType.DOUBLE), values.length);
        copy(handle, this, values, 0, 0, values.length);
    }

    /**
     * Writes the raster to this DArray in column major order.
     *
     * @param handle
     * @param raster
     */
    public DArray(Handle handle, Raster raster) {
        this(Array.empty(raster.getWidth() * raster.getHeight(), PrimitiveType.DOUBLE), raster.getWidth() * raster.getHeight());
//        raster.gets
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray copy(Handle handle) {
        DArray copy = DArray.empty(length);
        get(handle, copy, 0, 0, length);
        return copy;
    }

    /**
     * Constructs an array with a given GPU pointer and length.
     *
     * @param p A pointer to the first element of the array on the GPU.
     * @param length The length of the array.
     */
    protected DArray(CUdeviceptr p, int length) {
        super(p, length, PrimitiveType.DOUBLE);
    }

    /**
     * Creates an empty DArray with the specified size.
     *
     * @param size The number of elements in the array.
     * @return A new DArray with the specified size.
     * @throws ArrayIndexOutOfBoundsException if size is negative.
     */
    public static DArray empty(int size) {
        checkPositive(size);
        return new DArray(Array.empty(size, PrimitiveType.DOUBLE), size);
    }

    /**
     * Copies contents from a CPU array to a GPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param to The destination GPU array.
     * @param fromArray The source CPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param length The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public static void copy(Handle handle, DArray to, double[] fromArray, int toIndex, int fromIndex, int length) {
        checkNull(fromArray, to);

        Array.copy(
                handle,
                to,
                Pointer.to(fromArray),
                toIndex,
                fromIndex,
                length,
                PrimitiveType.DOUBLE
        );
    }

    /**
     * A pointer to a singleton array containing d.
     *
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    public static Pointer cpuPointer(double d) {
        return Pointer.to(new double[]{d});
    }
    
    /**
     * Copies the contents of this GPU array to a CPU array.
     *
     * @param to The destination CPU array.
     * @param toStart The index in the destination array to start copying to.
     * @param fromStart The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void get(Handle handle, double[] to, int toStart, int fromStart, int length) {
        checkNull(to);
        get(handle, Pointer.to(to), toStart, fromStart, length);
    }

    /**
     * Exports a portion of this GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param fromStart The starting index in this GPU array.
     * @param length The number of elements to export.
     * @return A CPU array containing the exported portion.
     * @throws IllegalArgumentException if fromStart or length is out of bounds.
     */
    public double[] get(Handle handle, int fromStart, int length) {
        double[] export = new double[length];
        handle.synch();
        get(handle, export, 0, fromStart, length);
        return export;
    }

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @return A CPU array containing all elements of this GPU array.
     */
    public double[] get(Handle handle) {
        return get(handle, 0, length);
    }

    /**
     * Copies from this vector to another with increments.
     *
     * @param handle handle to the cuBLAS library context.
     * @param to The array to copy to.
     * @param toStart The index to start copying to.
     * @param toInc stride between consecutive elements of the array copied to.
     * @param fromStart The index to start copying from.
     * @param fromInc stride between consecutive elements of this array.
     * @param length The number of elements to copy.
     */
    public void get(Handle handle, DArray to, int toStart, int fromStart, int toInc, int fromInc, int length) {

        int result = JCublas2.cublasDcopy(handle.get(),
                length,
                pointer(fromStart),
                fromInc,
                to.pointer(toStart),
                toInc
        );
        if (result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(result));
    }

    /**
     * Copies from this vector to another with increments.
     *
     * @param handle handle to the cuBLAS library context.
     * @param to The cpu array to copy to.
     * @param toStart The index to start copying to.
     * @param toInc stride between consecutive elements of the array copied to.
     * @param fromStart The index to start copying from.
     * @param fromInc stride between consecutive elements of this array.
     * @param length The number of elements to copy.
     */
    public void get(Handle handle, double[] to, int toStart, int fromStart, int toInc, int fromInc, int length) {
        if (fromInc == toInc && fromInc == 1)
            get(handle, to, toStart, fromStart, length);
        else {
            for (int i = 0; i < length; i++)
                get(handle, to, i * toInc + toStart, i * fromInc + fromStart, 1);
        }
    }

    /**
     * Copies from to vector from another with increments.
     *
     * @param handle handle to the cuBLAS library context.
     * @param from The array to copy from.
     * @param fromStart The index to begin copying from.
     * @param toInc stride between consecutive elements of the array copied to.
     * @param toStart The index to begin copying to.
     * @param fromInc stride between consecutive elements of this array.
     * @param length The number of elements to copy.
     */
    public void set(Handle handle, DArray from, int toStart, int fromStart, int toInc, int fromInc, int length) {

        from.get(handle, this, toStart, fromStart, toInc, fromInc, length);
    }

    /**
     * Copies a CPU array to this GPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param from The source CPU array.
     * @param toIndex The index in this GPU array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param size The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or size is
     * negative.
     */
    public void set(Handle handle, double[] from, int toIndex, int fromIndex, int size) {
        copy(handle, this, from, toIndex, fromIndex, size);
    }

    /**
     * Copies a CPU array to this GPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param from The source CPU array.
     * @throws IllegalArgumentException if from is null.
     */
    public final void set(Handle handle, double[] from) {
        set(handle, from, 0, 0, from.length);
    }

    /**
     * Copies a CPU array to this GPU array starting from a specified index.
     *
     * @param handle The handle.
     * @param from The source CPU array.
     * @param toIndex The index in this GPU array to start copying to.
     * @throws IllegalArgumentException if from is null.
     */
    public void set(Handle handle, double[] from, int toIndex) {
        set(handle, from, toIndex, 0, from.length);
    }

    /**
     * A sub array of this array. Note, this is not a copy and changes to this
     * array will affect the sub array and vice versa.
     *
     * @param start The beginning of the sub array.
     * @param length The length of the sub array.
     * @return A sub Array.
     */
    public DArray subArray(int start, int length) {
        checkPositive(start, length);
        checkAgainstLength(start + length - 1, start);
        return new DArray(pointer(start), length);
    }

    /**
     * A sub array of this array. Note, this is not a copy and changes to this
     * array will affect the sub array and vice versa. The length of the new
     * array will go to the end of this array.
     *
     * @param start The beginning of the sub array.
     * @return A sub array.
     */
    public DArray subArray(int start) {
        checkPositive(start);
        return new DArray(pointer(start), length - start);
    }

    /**
     * Sets the value at the given index.
     *
     * @param handle handle to the cuBLAS library context.
     * @param index The index the new value is to be assigned to.
     * @param val The new value at the given index.
     */
    public void set(Handle handle, int index, double val) {
        checkPositive(index);
        checkAgainstLength(index);
        set(handle, new double[]{val}, index);
    }

    /**
     * Gets the value from the given index.
     *
     * @param index The index the value is to be retrieved from.
     * @return The value at index.
     */
    public DSingleton get(int index) {
        checkPositive(index);
        checkAgainstLength(index);
        return new DSingleton(this, index);
    }

    /**
     * Maps the elements in this array at the given increment to a double[].
     *
     * @param handle The handle.
     * @param inc The increment of the desired elements.
     * @return A cpu array of all the elements at the given increment.
     */
    public double[] getIncremented(Handle handle, int inc) {
        double[] incremented = new double[n(inc)];
        Pointer cpu = Pointer.to(incremented);
        for (int i = 0; i < incremented.length; i++)
            get(handle, cpu, i, i*inc, 1);
        handle.synch();
        return incremented;
    }

    /**
     * Performs the rank-1 update: This is outer product.
     *
     * <pre>
     * this = multProd * X * Y^T + this
     * </pre>
     *
     * Where X is a column vector and Y^T is a row vector.
     *
     * @param handle handle to the cuBLAS library context.
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param multProd Scalar applied to the outer product of X and Y^T.
     * @param vecX Pointer to vector X in GPU memory.
     * @param incX When iterating thought the elements of x, the jump size. To
     * read all of x, set to 1.
     * @param vecY Pointer to vector Y in GPU memory.
     * @param incY When iterating though the elements of y, the jump size.
     * @param lda The distance between the first element of each column of a.
     */
    public void outerProd(Handle handle, int rows, int cols, double multProd, DArray vecX, int incX, DArray vecY, int incY, int lda) {
        checkNull(handle, vecX, vecY);
        checkPositive(lda, cols);
        checkLowerBound(1, incY, incX);
        checkAgainstLength(lda * cols - 1);

        int result = JCublas2.cublasDger(handle.get(), rows, cols, cpuPointer(multProd), vecX.pointer, incX, vecY.pointer, incY, pointer, lda);
        if (result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(result));
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm): This method
     * synchronizes the handle.
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars that will be squared.
     * @param inc The stride step over this array.
     * @return The Euclidean norm of this vector.
     */
    public double norm(Handle handle, int length, int inc) {
        DSingleton result = new DSingleton();
        norm(handle, length, inc, result);
        return result.getVal(handle);
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm):
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars that will be squared.
     * @param inc The stride step over this array.
     * @param result where the result is to be stored.
     */
    public void norm(Handle handle, int length, int inc, DSingleton result) {

        checkNull(handle, result);
        int errorCode = JCublas2.cublasDnrm2(handle.get(), length, pointer, inc, result.pointer);
        if (errorCode != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(errorCode));
    }

    /**
     * Finds the index of the element with the minimum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @param result where the result (index of the minimum absolute value) is
     * to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void argMinAbs(Handle handle, int length, int inc, int[] result, int toIndex) {
        checkNull(handle, result);
        int error = JCublas2.cublasIdamin(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.INT));
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        result[toIndex] -= 1;//It looks like the cuda methods are index-1 based.
    }

    /**
     * Finds the index of the element with the maximum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @param result where the result (index of the maximum absolute value) is
     * to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void argMaxAbs(Handle handle, int length, int inc, int[] result, int toIndex) {
        checkNull(handle, result);
        int error = JCublas2.cublasIdamax(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.INT));
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        result[toIndex] -= 1; //It looks like the cuda methods are index-1 based.
    }

    /**
     * Finds the index of the element with the minimum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @return The index of the lement with the minimum absolute value.
     */
    public int argMinAbs(Handle handle, int length, int inc) {
        int[] result = new int[1];
        argMinAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Finds the index of the element with the maximum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of max(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @return The index of the element with greatest absolute value.
     *
     */
    public int argMaxAbs(Handle handle, int length, int inc) {
        int[] result = new int[1];
        argMaxAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the sum of the absolute values of the vector X (1-norm):
     *
     * <pre>
     * result = |X[0]| + |X[1]| + ... + |X[n-1]|
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars to include in the sum.
     * @param inc The stride step over the array.
     * @return The l1 norm of the vector.
     */
    public double sumAbs(Handle handle, int length, int inc) {
        double[] result = new double[1];
        sumAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the sum of the absolute values of the vector X (1-norm):
     *
     * <pre>
     * result = |X[0]| + |X[1]| + ... + |X[n-1]|
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars to include in the sum.
     * @param inc The stride step over the array.
     * @param result where the result is to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void sumAbs(Handle handle, int length, int inc, double[] result, int toIndex) {
        checkNull(handle, result);
        int error = JCublas2.cublasDasum(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.DOUBLE));
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * Performs the matrix-vector multiplication:
     *
     * <pre>
     * this = timesAx * op(A) * X + beta * this
     * </pre>
     *
     * Where op(A) can be A or its transpose.
     *
     * @param handle handle to the cuBLAS library context.
     * @param transA Specifies whether matrix A is transposed (true for
     * transpose and false for not.)
     * @param aRows The number of rows in matrix A.
     * @param aCols The number of columns in matrix A.
     * @param timesAx Scalar multiplier applied to the matrix-vector product.
     * @param matA Pointer to matrix A in GPU memory.
     * @param lda The distance between the first element of each column of A.
     * @param vecX Pointer to vector X in GPU memory.
     * @param incX The increments taken when iterating over elements of X. This
     * is usually1 1. If you set it to 2 then you'll be looking at half the
     * elements of x.
     * @param beta Scalar multiplier applied to vector Y before adding the
     * matrix-vector product.
     * @param inc the increment taken when iterating over elements of this
     * array.
     * @return this array after this = timesAx * op(A) * X + beta*this
     */
    public DArray addProduct(Handle handle, boolean transA, int aRows, int aCols, double timesAx, DArray matA, int lda, DArray vecX, int incX, double beta, int inc) {
        checkNull(handle, matA, vecX);
        checkPositive(aRows, aCols);
        checkLowerBound(1, inc, incX);
        matA.checkAgainstLength(aRows * aCols - 1);

        int error = JCublas2.cublasDgemv(
                handle.get(),
                transpose(transA),
                aRows, aCols,
                cpuPointer(timesAx),
                matA.pointer, lda,
                vecX.pointer, incX,
                cpuPointer(beta),
                pointer,
                inc
        );
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        return this;
    }

    /**
     * Multiplies this vector by a banded matrix and adds the result to the
     * vector.
     *
     * this = timesAx * op(A) * X + timesThis * this
     *
     * A banded matrix is a sparse matrix where the non-zero elements are
     * confined to a diagonal band, comprising the main diagonal, a fixed number
     * of subdiagonals below the main diagonal, and a fixed number of
     * superdiagonals above the main diagonal. Banded matrices are often used to
     * save space, as only the elements within the band are stored, and the rest
     * are implicitly zero.
     *
     * In this method, the banded matrix is represented by the {@link DArray} M,
     * and the structure of M is defined by the number of subdiagonals and
     * superdiagonals. The matrix is stored in a column-major order, with each
     * column being a segment of the band. The parameter `lda` indicates the
     * leading dimension of the banded matrix, which corresponds to the number
     * of rows in the compacted matrix representation. The elements of the band
     * are stored contiguously in memory, with zero-padding where necessary to
     * fill out the bandwidth of the matrix.
     *
     * Let M represent the column-major matrix that stores the elements of A in
     * a {@link DArray}. The first row of M corresponds to the top-rightmost
     * non-zero diagonal of A (the highest superdiagonal). The second row
     * corresponds to the diagonal that is one position below/left of the first
     * row, and so on, proceeding down the diagonals. The final row of M
     * contains the bottom-leftmost diagonal of A (the lowest subdiagonal).
     * Diagonals that do not fully extend across A are padded with zeros in M.
     * An element in A has the same column in M as it does in A.
     *
     * This method performs the matrix-vector multiplication between the banded
     * matrix A and the vector x using the JCublas `cublasDgbmv` function, which
     * supports operations on banded matrices. The result is scaled by `timesA`
     * and added to this vector scaled by `timesThis`.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param transposeA Whether to transpose matrix A before multiplying.
     * @param rowsA The number of rows in matrix A.
     * @param colsA The number of columns in matrix A.
     * @param subDiagonalsA The number of subdiagonals in matrix A.
     * @param superDiagonalA The number of superdiagonals in matrix A.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray} representing the banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted banded matrix representation.
     * @param x The {@link DArray} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray addProductBandMatVec(Handle handle, boolean transposeA, int rowsA, int colsA, int subDiagonalsA, int superDiagonalA, double timesA, DArray Ma, int ldm, DArray x, int incX, double timesThis, int inc) {
        int error = JCublas2.cublasDgbmv(
                handle.get(),
                transpose(transposeA),
                rowsA, colsA,
                subDiagonalsA, superDiagonalA,
                cpuPointer(timesA),
                Ma.pointer, ldm,
                x.pointer, incX,
                cpuPointer(timesThis), pointer, inc);
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));

        return this;
    }

    /**
     * Solves the system of linear equations Ax = b, where A is a triangular
     * banded matrix, and x is the solution vector.
     *
     * b is this when the algorithm begins, and x is this when the algorithm
     * ends. That is to say the solution to Ax = this is stored in this.
     *
     * A triangular banded matrix is a special type of sparse matrix where the
     * non-zero elements are confined to a diagonal band around the main
     * diagonal and only the elements above (or below) the diagonal are stored,
     * depending on whether the matrix is upper or lower triangular.
     *
     * The matrix is stored in a compact banded format, where only the diagonals
     * of interest are represented to save space. For a lower diagonal matrix,
     * the first row represents the main diagonal, and subsequent rows represent
     * the diagonals progressively further from the main diagonal. An upper
     * diagonal matrix is stored with the last row as the main diagonal and the
     * first row as the furthest diagonal from the main.
     *
     * This method uses JCublas `cublasDtbsv` function to solve the system of
     * equations for the vector x.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param isUpper Indicates whether the matrix A is upper or lower
     * triangular. Use {@code cublasFillMode.CUBLAS_FILL_MODE_UPPER} for upper
     * triangular, or {@code cublasFillMode.CUBLAS_FILL_MODE_LOWER} for lower
     * triangular.
     * @param transposeA Whether to transpose the matrix A before solving the
     * system. Use {@code cublasOperation.CUBLAS_OP_T} for transpose, or
     * {@code cublasOperation.CUBLAS_OP_N} for no transpose.
     * @param onesOnDiagonal Specifies whether the matrix A is unit triangular
     * ({@code cublasDiagType.CUBLAS_DIAG_UNIT}) or non-unit triangular
     * ({@code cublasDiagType.CUBLAS_DIAG_NON_UNIT}).
     * @param rowsA The number of rows/columns of the matrix A (the order of the
     * matrix).
     * @param nonPrimaryDiagonals The number of subdiagonals or superdiagonals
     * in the triangular banded matrix.
     * @param Ma A compact form {@link DArray} representing the triangular
     * banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted matrix representation.
     *
     * @param inc The stride for stepping through the elements of b.
     * @return The updated {@link DArray} (b), now containing the solution
     * vector x.
     */
    public DArray solveTriangularBandedSystem(Handle handle, boolean isUpper, boolean transposeA, boolean onesOnDiagonal, int rowsA, int nonPrimaryDiagonals, DArray Ma, int ldm, int inc) {
        // Call the cublasDtbsv function to solve the system
        int error = JCublas2.cublasDtbsv(
                handle.get(),
                isUpper ? cublasFillMode.CUBLAS_FILL_MODE_UPPER : cublasFillMode.CUBLAS_FILL_MODE_LOWER, // Upper or lower triangular matrix
                transpose(transposeA),
                onesOnDiagonal ? cublasDiagType.CUBLAS_DIAG_UNIT : cublasDiagType.CUBLAS_DIAG_NON_UNIT, // Whether A is unit or non-unit triangular
                rowsA, // Number of rows/columns in A
                nonPrimaryDiagonals, // Number of subdiagonals/superdiagonals
                Ma.pointer, // Pointer to the compact form of matrix A
                ldm, // Leading dimension of Ma
                pointer, // Pointer to the right-hand side vector (b)
                inc);          // Stride through the elements of b
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
        return this;  // The result (solution vector x) is stored in b
    }

    @Override
    public String toString() {
        JCuda.cudaDeviceSynchronize();
        try (Handle handle = new Handle()) {
            return Arrays.toString(get(handle));
        }

    }

    /**
     * Multiplies this vector by a symmetric banded matrix and adds the result
     * to the vector.
     *
     * this = timesA * A * x + timesThis * this
     *
     * A symmetric banded matrix is a matrix where the non-zero elements are
     * confined to a diagonal band around the main diagonal, and the matrix is
     * symmetric (i.e., A[i][j] = A[j][i]). In a symmetric banded matrix, only
     * the elements within the band are stored, as the symmetry allows the upper
     * or lower part to be inferred. This storage technique reduces memory
     * usage.
     *
     * In this method, the symmetric banded matrix is represented by the A
     * stored in Ma, where only the upper (or lower) part of the matrix is
     * stored. The matrix is stored in a compact form, with each column being a
     * segment of the band. The parameter `ldm` indicates the leading dimension
     * of the matrix, which corresponds to the number of rows in the compacted
     * matrix representation. Only the non-zero diagonals of the matrix are
     * stored contiguously in memory.
     *
     * Let M represent the column-major matrix that stores the elements of the
     * symmetric banded matrix A in a {@link DArray}. The first row of M
     * corresponds to the main diagonal of A, and the subsequent rows correspond
     * to diagonals above or below the main diagonal. For instance, the second
     * row corresponds to the diagonal directly above the main diagonal, and so
     * on.
     *
     * This method performs the matrix-vector multiplication between the
     * symmetric banded matrix A and the vector x using the JCublas
     * `cublasDsbmv` function, which supports operations on symmetric banded
     * matrices. The result is scaled by `timesA` and added to this vector
     * scaled by `timesThis`.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param upper Whether the upper triangular part of the matrix is stored.
     * @param colA The order of the symmetric matrix A (number of rows and
     * columns).
     * @param diagonals The number of subdiagonals or superdiagonals in the
     * matrix.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray} representing the symmetric banded
     * matrix A.
     * @param ldm The leading dimension of the matrix, defining the row count in
     * the compacted matrix representation.
     * @param x The {@link DArray} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray addProductSymBandMatVec(Handle handle, boolean upper, int colA, int diagonals, double timesA, DArray Ma, int ldm, DArray x, int incX, double timesThis, int inc) {
        int error = JCublas2.cublasDsbmv(
                handle.get(),
                upper ? cublasFillMode.CUBLAS_FILL_MODE_UPPER : cublasFillMode.CUBLAS_FILL_MODE_LOWER,
                colA,
                diagonals,
                cpuPointer(timesA),
                Ma.pointer, ldm,
                x.pointer,
                incX,
                cpuPointer(timesThis),
                pointer,
                inc);
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));

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
     * @return this;
     */
    public DArray fill(Handle handle, double fill, int inc) {
        checkPositive(inc);
        checkNull(handle);

        DSingleton from = new DSingleton(handle, fill);
        set(handle, from, 0, 0, inc, 0, n(inc));
        return this;
    }

    /**
     * Fills a matrix with a value.
     *
     * @param handle handle to the cuBLAS library context.
     * @param height The height of the matrix.
     * @param width The width of the matrix.
     * @param lda The distance between the first element of each column of the
     * matrix. This should be at least the height of the matrix.
     * @param fill The value the matrix is to be filled with.
     * @return this, after having been filled.
     */
    public DArray fillMatrix(Handle handle, int height, int width, int lda, double fill) {
        checkPositive(height, width);
        checkLowerBound(height, lda);
        checkAgainstLength(height * width - 1);

        if (height == lda) {
            if (fill == 0) {
                subArray(0, width * height).fill0(handle);
                return this;
            }
            return subArray(0, width * height).fill(handle, fill, 1);
        }

        try (DArray filler = new DSingleton(handle, fill)) {
            int size = height * width;
            KernelManager kern = KernelManager.get("fillMatrix");
            kern.map(handle, filler, lda, this, height, size);
        }

        return this;
    }

    public static void main(String[] args) {
        try (
                Handle hand = new Handle();
                DArray d = new DArray(hand, 1, 2, 3, 4, 5, 6);
                DArray x = empty(6).fill(hand, 2, 1)) {

            d.add(hand, 3, x, 2, 2);

            System.out.println(d);
        }
    }

    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param incX The number of spaces to jump when incrementing forward
     * through x.
     * @param inc The number of spaces to jump when incrementing forward through
     * this array.
     * @param x Pointer to vector X in GPU memory.
     * @return The dot product of X and Y.
     */
    public double dot(Handle handle, DArray x, int incX, int inc) {
        double[] result = new double[1];
        dot(handle, x, incX, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param incX The number of spaces to jump when incrementing forward
     * through x.
     * @param inc The number of spaces to jump when incrementing forward through
     * this array.
     * @param x Pointer to vector X in GPU memory.
     * @param result The array the answer should be put in.
     * @param resultInd The index of the array the answer should be put in.
     */
    public void dot(Handle handle, DArray x, int incX, int inc, double[] result, int resultInd) {
        checkNull(handle, x, result);
        checkPositive(resultInd, inc, incX);
        int error = JCublas2.cublasDdot(handle.get(), n(inc), x.pointer, incX, pointer, inc, Pointer.to(result).withByteOffset(resultInd * Sizeof.DOUBLE));
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));

    }

    /**
     * Performs the matrix-matrix multiplication using double precision (Dgemm)
     * on the GPU:
     *
     * <pre>
     * this = op(A) * op(B) + this
     * </pre>
     *
     * Where op(A) and op(B) represent A and B or their transposes based on
     * `transa` and `transb`.
     *
     * @param handle There should be one handle in each thread.
     * @param transposeA True opA should be transpose, false otherwise.
     * @param transposeB True if opB should be transpose, false otherwise.
     * @param aRows The number of rows of matrix C and matrix A (if
     * !transposeA).
     * @param bThisCols The number of columns of this matrix and matrix B (if
     * !transposeP).
     * @param aColsBRows The number of columns of matrix A (if !transposeA) and
     * rows of matrix B (if !transposeB).
     * @param timesAB A scalar to be multiplied by AB.
     * @param a Pointer to matrix A, stored in GPU memory. successive rows in
     * memory, usually equal to ARows).
     * @param lda The number of elements between the first element of each
     * column of A. If A is not a subset of a larger data set, then this will be
     * the height of A.
     * @param b Pointer to matrix B, stored in GPU memory.
     * @param ldb @see lda
     * @param timesCurrent This is multiplied by the current array first and
     * foremost. Set to 0 if the current array is meant to be empty, and set to
     * 1 to add the product to the current array as is.
     * @param ldc @see ldb
     */
    public void addProduct(Handle handle, boolean transposeA, boolean transposeB, int aRows,
            int bThisCols, int aColsBRows, double timesAB, DArray a, int lda, DArray b, int ldb, double timesCurrent, int ldc) {
        checkNull(handle, a, b);
        checkPositive(aRows, bThisCols, aColsBRows, lda, ldb, ldc);
        if (!transposeA) checkLowerBound(aRows, lda);
        if (!transposeB) checkLowerBound(aColsBRows);

        a.checkAgainstLength(aColsBRows * lda - 1);
        checkAgainstLength(aRows * bThisCols - 1);

        int error = JCublas2.cublasDgemm(
                handle.get(), // cublas handle
                transpose(transposeA), transpose(transposeB),
                aRows, bThisCols, aColsBRows,
                cpuPointer(timesAB),
                a.pointer, lda,
                b.pointer, ldb,
                cpuPointer(timesCurrent),
                pointer, ldc
        );
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * Performs the vector addition:
     *
     * <pre>
     * this = timesX * X + this
     * </pre>
     *
     * This operation scales vector X by alpha and adds it to vector Y.
     *
     * @param handle handle to the cuBLAS library context.
     * @param timesX Scalar used to scale vector X.
     * @param x Pointer to vector X in GPU memory.
     * @param incX The number of elements to jump when iterating forward through
     * x.
     * @param inc The number of elements to jump when iterating forward through
     * this.
     * @return this
     */
    public DArray add(Handle handle, double timesX, DArray x, int incX, int inc) {
        checkNull(handle, x);
        checkLowerBound(1, inc);
        checkAgainstLength((n(inc) - 1) * inc);
        checkMemAllocation();
        x.checkMemAllocation();
        
        if (incX != 0 && x.n(incX) != n(inc))
            throw new DimensionMismatchException(n(inc), x.n(incX));

        int result = JCublas2.cublasDaxpy(
                handle.get(),
                n(inc),
                cpuPointer(timesX), x.pointer, incX,
                pointer, inc
        );
        if (result != cudaError.cudaSuccess){
            throw new RuntimeException("cuda addition failed. Error: " + result + " - " + cudaError.stringFor(result) + "\n the arrays are:\n" + x.toString() + "\n with increment " +  incX 
                    + "and \n" + toString() + " with increment " + inc);
        }
        
        return this;
    }

    /**
     * Performs matrix addition or subtraction.
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
     * @param height number of rows of matrix C
     * @param width number of columns of matrix C
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param lda leading dimension of matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @param ldb leading dimension of matrix B
     * @param ldc leading dimension of matrix C
     * @return this
     *
     */
    public DArray setSum(Handle handle, boolean transA, boolean transB, int height,
            int width, double alpha, DArray a, int lda, double beta, DArray b,
            int ldb, int ldc) {
        checkNull(handle, a, b);
        checkPositive(height, width);
        checkAgainstLength(height * width - 1);
        checkMemAllocation();
        a.checkMemAllocation();

        int result = JCublas2.cublasDgeam(
                handle.get(),
                transpose(transA), transpose(transB),
                height, width,
                cpuPointer(alpha), a.pointer, lda,
                cpuPointer(beta), b.pointer, ldb,
                pointer, ldc
        );
        if (result != cudaError.cudaSuccess){
            throw new RuntimeException("cuda error " + result + " - " + cudaError.stringFor(result));
        }

        return this;
    }

    /**
     * Scales this vector by the scalar mult:
     *
     * <pre>
     * this = mult * this
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param mult Scalar multiplier applied to vector X.
     * @param inc The number of elements to jump when iterating forward through
     * this array.
     * @return this;
     *
     *
     */
    public DArray multiply(Handle handle, double mult, int inc) {
        checkNull(handle);
        checkLowerBound(1, inc);
        int result = JCublas2.cublasDscal(handle.get(), n(inc), cpuPointer(mult), pointer, inc);
        if (result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(result));
        return this;
    }

    /**
     * TODO: put a version of this in the matrix class.
     *
     * Performs symmetric matrix-matrix multiplication using.
     *
     * Computes this = A * A^T + timesThis * this, ensuring C is symmetric.
     *
     * @param handle CUBLAS handle for managing the operation.
     * @param transpose
     * @param uplo Specifies which part of the matrix is being used (upper or
     * lower).
     * @param resultRowsCols The number of rows/columns of the result matrices.
     * @param cols The number of columns of A (for C = A * A^T).
     * @param alpha Scalar multiplier for A * A^T.
     * @param a Pointer array to the input matrices.
     * @param lda Leading dimension of A.
     * @param timesThis Scalar multiplier for the existing C matrix (usually 0
     * for new computation).
     * @param ldThis Leading dimension of C.
     *
     */
    public void matrixSquared(
            Handle handle,
            boolean transpose,
            int uplo, // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
            int resultRowsCols,
            int cols,
            double alpha,
            DArray a,
            int lda,
            double timesThis,
            int ldThis) {

        int result = JCublas2.cublasDsyrk(
                handle.get(),
                uplo,
                transpose(transpose),
                resultRowsCols, cols,
                cpuPointer(alpha), a.pointer, lda,
                cpuPointer(alpha),
                pointer, ldThis
        );
        if (result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(result));
    }

    /**
     * Breaks this array into a a set of sub arrays.
     *
     * @param strideSize The length of each sub array.
     * @param batchSize The number of elements in the batch.
     * @param subArrayLength The number of elements in each subArray.
     * @return A representation of this array as a set of sub arrays.
     */
    public DStrideArray getAsBatch(int strideSize, int subArrayLength, int batchSize) {
        return new DStrideArray(pointer, strideSize, subArrayLength, batchSize);
    }

    /**
     * Breaks this array into a a set of sub arrays.
     *
     * @param handle
     * @param strideSize The length of each sub array, except for the last one
     * which may be longer.
     * @return A representation of this array as a set of sub arrays.
     */
    public DPointerArray getPointerArray(Handle handle, int strideSize) {
        DPointerArray dPoint;

        if (strideSize == 0) dPoint = DPointerArray.empty(1, strideSize);
        else dPoint = DPointerArray.empty(length / strideSize, strideSize);

        return dPoint.fill(handle, this, strideSize);
    }

    /**
     * The maximum number of times something can be done at the requested
     * increment.
     *
     * @param inc The increment between the elements that the something is done
     * with.
     * @return The number of times is can be done
     */
    private int n(int inc) {
        return Math.ceilDiv(length, inc);
    }

}
------------------------------------------------
package main;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.resourceManagement.Handle;
import java.util.function.IntFunction;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient implements AutoCloseable{

    private Matrix dX, dY;

    /**
     * Computes the gradients of an image in both the x and y directions.
     * Gradients are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     *
     */
    public Gradient(Matrix pic, Handle hand) {
        int width = pic.getWidth(), height = pic.getHeight();

        dX = new Matrix(hand, height, width);
        dY = new Matrix(hand, height, width);

        computeBoundaryGradients(i -> pic.getColumn(i), i -> dX.getColumn(i), width);
        computeBoundaryGradients(i -> pic.getRow(i), i -> dY.getRow(i), height);

        computeInteriorGradients(hand, pic, width, height, height, diff.length, dX.columns());
        computeInteriorGradients(hand, pic, height, 1, diff.length, width, dY.rows());
    }

    /**
     * Sets the border rows (columns) of the gradient matrices.
     *
     * Computes the gradients at the boundary of the image using forward and
     * backward differences. This method handles the first and last rows and
     * columns of the image, where gradients are calculated with one-sided
     * differences.
     *
     * The matrices from which pic and dM are taken should be pixel intensities.
     *
     * @param pic The picture the gradient is being taken from. The output of
     * this method should be the rows (columns) that the gradient is taken over.
     * @param dM Where the gradient is to be stored. The output should be the
     * rows (columns) where the gradient is stored. Use rows (columns) for dY
     * (dX).
     * @param length The max row (column) index exclusive.
     */
    private void computeBoundaryGradients(IntFunction<Matrix> pic, IntFunction<Matrix> dM, int length) {
        dM.apply(0).setSum(-1, pic.apply(0), 1, pic.apply(1));
        dM.apply(length - 1).setSum(-1, pic.apply(length - 2), 1, pic.apply(length - 1));
        dM.apply(1).setSum(-0.5, pic.apply(0), 0.5, pic.apply(2));
        dM.apply(length - 2).setSum(-0.5, pic.apply(length - 3), 0.5, pic.apply(length - 1));
    }

    /**
     * An array used for differentiation.
     */
    private static final DArray diff;

    static {
        try (Handle hand = new Handle()) {
            diff = new DArray(hand, 1.0 / 12, -2.0 / 3, 0, 2.0 / 3, -1.0 / 12);
        }
    }

    /**
     * Computes the gradients for the interior pixels using higher-order
     * differences. This method calculates the gradients for pixels that are not
     * near the boundary, using a higher-order finite difference scheme for
     * increased accuracy.
     *
     * The blocks are sub matrices of rows or columns that are added/subtracted
     * according to diff to find the gradient.
     *
     * @param hand The handle.
     * @param pic The picture over which the gradient is taken.
     * @param length Either the height or the width as appropriate.
     * @param blockStride This should be one if the blocks are made of rows
     * @param blockHeight The height of each block.  For row blocks this should be diff.length and for col blocks this should be height.
     * @param blockWidth see block height but opposite.
     * @param target Where the results are stored.  This should either be dX.columns() or dY.rows()
     */
    private void computeInteriorGradients(Handle hand, Matrix pic, int length, int blockStride, int blockHeight, int blockWidth, VectorsStride target) {

        // Interior x gradients (third column to second-to-last)
        int numBlocks = length - diff.length + 1;

        VectorsStride diffVecs = new VectorsStride(hand, diff, 1, diff.length, 0, numBlocks);

        MatricesStride blocks = new MatricesStride(hand, pic.dArray(), blockHeight, blockWidth, pic.colDist, blockStride, numBlocks);
        
        target = target.subBatch(2, numBlocks);

        if (blocks.height == diff.length) target.setProduct(diffVecs, blocks);
        else target.setProduct(blocks, diffVecs);
    }

    /**
     * An unmodifiable x gradient matrix.
     *
     * @return An unmodifiable x gradient matrix.
     */
    public Matrix x() {
        return dX;
    }

    /**
     * An unmodifiable y gradient matrix.
     *
     * @return An unmodifiable y gradient matrix.
     */
    public Matrix y() {
        return dY;
    }

    @Override
    public void close() {
        dX.close();
        dY.close();
    }

}
------------------------------------------------
package JCudaWrapper.algebra;

import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
import java.util.Arrays;
import org.apache.commons.math3.exception.DimensionMismatchException;
import JCudaWrapper.resourceManagement.Handle;
import java.awt.Dimension;

/**
 * This class provides methods for handling a batch of strided matrices stored
 * in row-major format. Each matrix in the batch can be accessed and operated on
 * individually or as part of the batch. The class extends {@code Matrix} and
 * supports operations such as matrix multiplication and addition. The strided
 * storage is designed to support JCuda cuSolver methods that require row-major
 * memory layout.
 *
 * Strided matrices are stored with a defined distance (stride) between the
 * first elements of each matrix in the batch.
 *
 * @author E. Dov Neimand
 */
public class MatricesStride implements ColumnMajor, AutoCloseable {

    protected final Handle handle;
    public final int height, width, colDist;
    protected final DStrideArray data;

    /**
     * Constructor for creating a batch of strided matrices. Each matrix is
     * stored with a specified stride and the batch is represented as a single
     * contiguous block in memory.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param height The number of rows (height) in each submatrix.
     * @param width The number of columns (width) in each submatrix.
     * @param stride The number of elements between the first elements of
     * consecutive submatrices in the batch.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int height, int width, int stride, int batchSize) {
        this(
                handle,
                DArray.empty(DStrideArray.totalDataLength(stride, width * height, batchSize)),
                height,
                width,
                height,
                stride,
                batchSize
        );
    }

    /**
     * Creates a simple batch matrix with coldDist = height.
     *
     * @param handle
     * @param data The length of this data should be width*height.
     * @param height The height of this matrix and the sub matrices.
     * @param width The width of each sub matrix.
     * @param colDist The distance between the first element of each column.
     * @param strideSize How far between the first elements of each matrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, DArray data, int height, int width, int colDist, int strideSize, int batchSize) {
        this.handle = handle;
        this.data = data.getAsBatch(strideSize, colDist * (width - 1) + height, batchSize);
        this.height = height;
        this.width = width;
        this.colDist = colDist;
    }

    /**
     * Constructor for creating a batch of square strided matrices. Each matrix
     * is stored with a specified stride and the batch is represented as a
     * single contiguous block in memory. These matrices have no overlap.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int subHeight, int batchSize) {
        this(handle, subHeight, subHeight, batchSize);
    }

    /**
     * Constructor for creating a batch of square strided matrices. Each matrix
     * is stored with a specified stride and the batch is represented as a
     * single contiguous block in memory. These matrices have no overlap.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param height The number of rows (height) in each submatrix.
     * @param width The number of columns (width) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int height, int width, int batchSize) {
        this(handle, height, width, height * width, batchSize);
    }

    /**
     * Returns a vector of elements corresponding to row {@code i} and column
     * {@code j} across all matrices in the batch. Each element in the vector
     * corresponds to the element at position (i, j) in a different submatrix.
     *
     * @param i The row index in each submatrix.
     * @param j The column index in each submatrix.
     * @return A vector containing the elements at (i, j) for each submatrix in
     * the batch.
     */
    public Vector get(int i, int j) {
        return new Vector(handle, data.subArray(index(i, j)), data.stride);
    }

    /**
     * Retrieves all elements from each submatrix in the batch as a 2D array of
     * {@code Vector} objects. Each vector contains elements corresponding to
     * the same row and column across all matrices in the batch. The method
     * returns a 2D array where each element (i, j) is a vector of the elements
     * at position (i, j) from each matrix in the batch. "i" is the row and "j"
     * is the column.
     *
     * @return A 2D array of {@code Vector} objects. Each {@code Vector[i][j]}
     * represents the elements at row {@code i} and column {@code j} for all
     * submatrices in the batch.
     */
    public Vector[][] parition() {
        Vector[][] all = new Vector[height][width];
        for (int i = 0; i < height; i++) {
            int row = i;
            Arrays.setAll(all[row], col -> get(row, col));
        }
        return all;
    }

    /**
     * Performs matrix multiplication on the batches of matrices, and adds them
     * to this matrix. This method multiplies matrix batches {@code a} and
     * {@code b}, scales the result by {@code timesAB}, scales the existing
     * matrix in the current instance by {@code timesResult}, and then adds them
     * together and palces the result here.
     *
     * @param transposeA Whether to transpose the matrices in {@code a}.
     * @param transposeB Whether to transpose the matrices in {@code b}.
     * @param a The left-hand matrix batch in the multiplication.
     * @param b The right-hand matrix batch in the multiplication.
     * @param timesAB The scaling factor applied to the matrix product
     * {@code a * b}.
     * @param timesThis The scaling factor applied to the result matrix.
     * @return this
     * @throws DimensionMismatchException if the dimensions of matrices
     * {@code a} and {@code b} are incompatible for multiplication.
     */
    public MatricesStride addProduct(boolean transposeA, boolean transposeB, double timesAB, MatricesStride a, MatricesStride b, double timesThis) {

        Dimension aDim = new Dimension(transposeA ? a.height : a.width, transposeA ? a.width : a.height);
        Dimension bDim = new Dimension(transposeB ? b.height : b.width, transposeB ? b.width : b.height);

        if (aDim.width != bDim.height || height != aDim.height || width != bDim.width)
            throw new DimensionMismatchException(aDim.width, bDim.height);

        data.addProduct(handle, transposeA, transposeB,
                aDim.height, aDim.width, bDim.width,
                timesAB,
                a.data, a.colDist,
                b.data, b.colDist,
                timesThis, colDist
        );

        return this;
    }

    /**
     * Performs matrix-scalar multiplication on the batches of matrices, and
     * adds them to this matrix. This method multiplies matrix batches {@code a}
     * and {@code b}, scales the result by {@code timesAB}, scales the existing
     * matrix in the current instance by {@code timesResult}, and then adds them
     * together and palces the result here.
     *
     * @param scalars the ith element is multiplied by the ith matrix.
     * @return this
     * @throws DimensionMismatchException if the dimensions of matrices
     * {@code a} and {@code b} are incompatible for multiplication.
     */
    public MatricesStride multiply(Vector scalars) {

        KernelManager.get("prodScalarMatrixBatch").vectorBatchMatrix(handle, scalars, this);

        return this;
    }

    /**
     * Multiplies all the matrices in this set by the given scalar.
     *
     * @param scalar multiply all these matrices by this scalar.
     * @param workSpace Should be width * height.
     * @return this
     */
    public MatricesStride multiply(double scalar, DArray workSpace) {
        add(false, scalar, this, 0, workSpace);
        return this;
    }

    /**
     * Computes the eigenvalues. This batch must be a set of symmetric 2x2
     * matrices.
     *
     * @param workSpace Should be at least as long as batchSize.
     * @return The eigenvalues.
     */
    public VectorsStride computeVals2x2(Vector workSpace) {

        if (height != 2)
            throw new IllegalArgumentException("compute vals 2x2 can only be called on a 2x2 matrix.  These matrices are " + height + "x" + width);

        VectorsStride eVals = new VectorsStride(handle, 2, getBatchSize(), 2, 1);
        Vector[] eVal = eVals.vecPartition();//val[0] is the first eigenvalue, val[1] the seond, etc...

        Vector[][] m = parition();

        Vector trace = workSpace.getSubVector(0, data.batchCount());

        trace.set(m[1][1]).add(1, m[0][0]);//= a + d

        eVal[1].ebeSetProduct(m[0][1], m[0][1]).ebeAddProduct(m[0][0], m[1][1], -1);// = ad - c^2
        
        eVal[0].ebeSetProduct(trace, trace).add(-4, eVal[1]);//=(d + a)^2 - 4(ad - c^2)

        KernelManager.get("sqrt").mapToSelf(handle, eVal[0]);//sqrt((d + a)^2 - 4(ad - c^2))

        eVal[1].set(trace);
        eVal[1].add(-1, eVal[0]);
        eVal[0].add(1, trace);

        eVals.data.multiply(handle, 0.5, 1);

        return eVals;
    }

    /**
     * Computes the eigenvalues for a set of symmetric 3x3 matrices. If this
     * batch is not such a set then this method should not be called.
     *
     * @param workSpace Should have length equal to the width of this matrix.
     * @return The eigenvalues.
     *
     */
    //m := a, d, g, d, e, h, g, h, i = m00, m10, m20, m01, m11, m21, m02, m12, m22
    //p := tr m = a + e + i
    //q := (p^2 - norm(m)^2)/2 where norm = a^2 + d^2 + g^2 + d^2 + ...
    // solve: lambda^3 - p lambda^2 + q lambda - det m        
    public VectorsStride computeVals3x3(Vector workSpace) {

        if (height != 3)
            throw new IllegalArgumentException("computeVals3x3 can only be called on a 3x3 matrix.  These matrices are " + height + "x" + width);

        VectorsStride vals = new VectorsStride(handle, height, getBatchSize(), height, 1);
        Vector[] work = workSpace.parition(3);

        Vector[][] m = parition();

        Vector[][] minor = new Vector[3][3];

        Vector negTrace = negativeTrace(work[0]);//val[0] is taken, but val[1] is free.

        setDiagonalMinors(minor, m, vals);
        Vector C = work[1].fill(0);
        for (int i = 0; i < 3; i++) C.add(1, minor[i][i]);

        setRow0Minors(minor, m, vals);
        Vector det = work[2];
        det.ebeSetProduct(m[0][0], minor[0][0]);
        det.addEbeProduct(-1, m[0][1], minor[0][1], 1);
        det.addEbeProduct(-1, m[0][2], minor[0][2], -1);

        cubicRoots(negTrace, C, det, vals);

        return vals;
    }

    private static DArray negOnes3;

    /**
     * A vector of 3 negative ones.
     *
     * @return A vector of 3 negative ones.
     */
    private static VectorsStride negOnes3(Handle handle, int batchSize) {
        if (negOnes3 == null) negOnes3 = DArray.empty(3).fill(handle, -1, 1);
        return new VectorsStride(handle, negOnes3, 1, 3, 0, batchSize);
    }

    /**
     * The negative of the trace of the submatrices.
     *
     * @param traceStorage The vector that gets overwritten with the trace.
     * Should have batch elements.
     * @param ones a vector that will have -1's stored in it. It should have
     * height number of elements in it.
     * @return The trace.
     */
    private Vector negativeTrace(Vector traceStorage) {//TODO:Retest!

        VectorsStride diagnols = new VectorsStride(handle, data, 4, 3, data.stride, data.batchSize);

        return traceStorage.setBatchVecVecMult(
                diagnols,
                negOnes3(handle, data.batchSize)
        );

    }

    /**
     * Sets the minors of the diagonal elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param minorStorage A space where the minors can be stored.
     */
    private void setDiagonalMinors(Vector[][] minor, Vector[][] m, VectorsStride minorStorage) {

        Vector[] storagePartition = minorStorage.vecPartition();

        for (int i = 0; i < minor.length; i++)
            minor[i][i] = storagePartition[i];

        minor[0][0].ebeSetProduct(m[1][1], m[2][2]);
        minor[0][0].addEbeProduct(-1, m[1][2], m[1][2], 1);

        minor[1][1].ebeSetProduct(m[0][0], m[2][2]);
        minor[1][1].addEbeProduct(-1, m[0][2], m[0][2], 1);

        minor[2][2].ebeSetProduct(m[0][0], m[1][1]);
        minor[2][2].addEbeProduct(-1, m[0][1], m[0][1], 1);
    }

    /**
     * Sets the minors of the first row of elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param minorStorage A space where the minors can be stored.
     */
    private void setRow0Minors(Vector[][] minor, Vector[][] m, VectorsStride minorStorage) {
        minor[0] = minorStorage.vecPartition();

        minor[0][1].ebeSetProduct(m[1][1], m[2][2]);
        minor[0][1].addEbeProduct(-1, m[1][2], m[1][2], 1);

        minor[0][1].ebeSetProduct(m[0][1], m[2][2]);
        minor[0][1].addEbeProduct(-1, m[0][2], m[1][2], 1);

        minor[0][2].ebeSetProduct(m[0][1], m[1][2]);
        minor[0][2].addEbeProduct(-1, m[1][1], m[0][2], 1);
    }

    /**
     * Computes the real roots of a cubic equation in the form: x^3 + b x^2 + c
     * x + d = 0
     *
     * TODO: Since this method calls multiple kernels, it would probably be
     * faster if written as a single kernel.
     *
     * @param b Coefficients of the x^2 terms.
     * @param c Coefficients of the x terms.
     * @param d Constant terms.
     * @param roots An array of Vectors where the roots will be stored.
     */
    private static void cubicRoots(Vector b, Vector c, Vector d, VectorsStride roots) {
        KernelManager cos = KernelManager.get("cos"),
                acos = KernelManager.get("acos"),
                sqrt = KernelManager.get("sqrt");

        Vector[] root = roots.vecPartition();

        Vector q = root[0];
        q.ebeSetProduct(b, b);
        q.addEbeProduct(2.0 / 27, q, b, 0);
        q.addEbeProduct(-1.0 / 3, b, c, 1);
        q.add(1, d);

        Vector p = d;
        p.addEbeProduct(1.0 / 9, b, b, 0);
        p.add(-1.0 / 3, c); //This is actually p/-3 from wikipedia.

        //c is free for now.  
        Vector theta = c;
        Vector pInverse = root[1].fill(1).ebeDivide(p); //c is now taken               
        sqrt.map(b.getHandle(), pInverse, theta);

        theta.addEbeProduct(-0.5, q, theta, 0);//root[0] is now free (all roots).
        theta.ebeSetProduct(theta, pInverse); //c is now free.
        acos.mapToSelf(b.getHandle(), theta);

        for (int k = 0; k < 3; k++) root[k].set(theta).add(-2 * Math.PI * k);

        roots.data.multiply(b.getHandle(), 1.0 / 3, 1);
        cos.mapToSelf(b.getHandle(), roots.data);

        sqrt.mapToSelf(b.getHandle(), p);
        for (Vector r : root) {
            r.addEbeProduct(2, p, r, 0);
            r.add(-1.0 / 3, b);
        }
    }

    /**
     * The ith column of each submatrix.
     *
     * @param i The index of the desired column.
     * @return The ith column of each submatrix.
     */
    public VectorsStride column(int i) {
        return new VectorsStride(handle, data.subArray(i * colDist), 1, height, data.stride, data.batchSize);
    }

    /**
     * The ith row of each submatrix.
     *
     * @param i The index of the desired column.
     * @return The ith column of each submatrix.
     */
    public VectorsStride row(int i) {
        return new VectorsStride(handle, data.subArray(i), colDist, width, data.stride, data.batchSize);
    }

    /**
     * Partitions these matrices by column.
     *
     * @return This partitioned by columns.
     */
    public VectorsStride[] columnPartition() {
        VectorsStride[] patritioned = new VectorsStride[width];
        Arrays.setAll(patritioned, i -> column(i));
        return patritioned;

    }

    /**
     * Adds dimensions like batchsize and width to the given data.
     *
     * @param addDimensions data in need of batch dimensions.
     * @return The given data with this's dimensions.
     */
    public MatricesStride copyDimensions(DArray addDimensions) {
        return new MatricesStride(
                handle,
                addDimensions,
                height,
                width,
                colDist,
                data.stride,
                data.batchSize
        );
    }

    /**
     * Adds dimensions like batchsize and width to the given data.
     *
     * Stride and batch size are taken from add Dimensions, the rest of the
     * dimensions from this.
     *
     * @param addDimensions data in need of batch dimensions.
     * @return The given data with this's dimensions.
     */
    public MatricesStride copyDimensions(DStrideArray addDimensions) {
        return new MatricesStride(
                handle,
                addDimensions,
                height,
                width,
                colDist,
                addDimensions.stride,
                addDimensions.batchSize
        );
    }

    /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite. TODO: If any extra memory is available, pass
     * it here!
     *
     * @param eValues The eigenvalues, organized by sets per matrix.
     * @param workSpaceArray Should have as many elements as there are in this.
     * @return The eigenvectors.
     *
     */
    public MatricesStride computeVecs(VectorsStride eValues, DArray workSpaceArray) {

        MatricesStride eVectors = copyDimensions(DArray.empty(data.length));

        try (IArray pivot = IArray.empty(data.batchSize * height)) {

            for (int i = 0; i < height; i++) {
                MatricesStride workSpace = copy(workSpaceArray);//TODO: with each iteration, only elements on the diagonal change.  Why recopy the whole thing?
                workSpace.computeVec(eValues.getElement(i), eVectors.column(i), pivot);
            }
        }

        return eVectors;
    }

    /**
     * Computes an eigenvector for this matrix. This matrix will be changed.
     *
     * @param eValue The eigenvalues.
     * @param eVector Where the eigenvector will be placed.
     * @param info The success of the computations.
     */
    private void computeVec(Vector eValue, VectorsStride eVector, IArray pivot) {

        for (int i = 0; i < height; i++) get(i, i).add(-1, eValue);
        
        final double tolerance = 1e-10;
        
        KernelManager.get("nullSpace1dBatch").map(
                handle, 
                data, colDist, 
                eVector.data, eVector.getStrideSize(), 
                getBatchSize(), 
                IArray.cpuPointer(width), DArray.cpuPointer(tolerance), pivot.pointerToPointer()
         );

    }

    /**
     * Returns this matrix as a set of pointers.
     *
     * @return
     */
    public MatricesPntrs getPointers() {
        if (pointers == null) {
            pointers = new MatricesPntrs(
                    height, width, colDist, data.getPointerArray(handle)
            );
        }
        return pointers;
    }

    private MatricesPntrs pointers;

    /**
     * Returns this matrix as a set of pointers.
     *
     * @param putPointersHere An array where the pointers will be stored.
     * @return
     */
    public MatricesPntrs getPointers(DPointerArray putPointersHere) {

        return new MatricesPntrs(height, width, colDist, putPointersHere.fill(handle, data));

    }

    @Override
    public void close() {
        data.close();
    }

    /**
     * Gdets the matrix at the given index.
     *
     * @param i The index of the desired matrix.
     * @return The matrix at the requested index.
     */
    public Matrix getMatrix(int i) {
        return new Matrix(handle, data.getBatchArray(i), height, width, colDist);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < data.batchCount(); i++)
            sb.append(getMatrix(i).toString()).append("\n\n");

        return sb.toString();
    }

    /**
     * A copy of this matrices stride.
     *
     * @return A copy of this matrices stride.
     */
    public MatricesStride copy() {
        return copyDimensions(data.copy(handle));
    }

    /**
     * Copies from this matrix into the proffered matrix. Note, underlying data
     * is copied even if it does not appear in this matrix. TODO: fix this so
     * that unused underlying data is not copied.
     *
     * @param copyTo becomes a copy of this matrix.
     * @return the copy.
     */
    public MatricesStride copy(DArray copyTo) {
        copyTo.set(handle, data, 0, 0, data.length);

        return copyDimensions(copyTo);
    }

    public static void main(String[] args) {
        try (
                Handle handle = new Handle();
                DArray array = new DArray(handle, 1, 2, 2, 3, 9, 11, 11, 12, 5, 6, 6, 8, 1, 0, 0, 0)) {

            MatricesStride ms = new MatricesStride(handle, array, 2, 2, 2, 4, 4);

            Matrix[] m = new Matrix[4];
            Arrays.setAll(m, i -> new Matrix(handle, array.subArray(i * 4), 2, 2));
            for (int i = 0; i < m.length; i++) m[i].power(2);

            System.out.println("matrices:\n" + ms);


            try (Eigen eigen = new Eigen(ms)) {

                System.out.println("Eigen values:\n" + eigen.values);

                System.out.println("Eigen vectors:\n" + eigen.vectors);

                System.out.println("verifying:\n");
                for (int matInd = 0; matInd < ms.getBatchSize(); matInd++)
                    for (int j = 0; j < ms.height; j++)
                        System.out.println(ms.getMatrix(matInd) + " * " + eigen.vectors.getMatrix(matInd).getColumn(j) + " * " + "1/" + eigen.values.getVector(matInd).get(j) + " = "
                                + eigen.vectors.getMatrix(matInd).getColumn(j).addProduct(
                                        false,
                                        1 / eigen.values.getVector(matInd).get(j),
                                        ms.getMatrix(matInd),
                                        eigen.vectors.getMatrix(matInd).getColumn(j),
                                        0
                                ));
            }
        }
    }

    
//    public static void main(String[] args) {
//        try(Handle hand = new Handle(); DArray da = new DArray(hand, 1,0,0,0)){
//            Matrix m = new Matrix(hand, da, 2, 2);
//            
//            System.out.println("matrix = \n" + m + "\n");
//            
//            Eigen eigen = new Eigen(m.repeating(1));
//            
//            System.out.println("values: " + eigen.values + "\n");
//            System.out.println("vectors:\n" + eigen.vectors);
//            
//        }
//    }
    
    /**
     * The underlying batch array.
     *
     * @return The underlying batch arrays.
     */
    public DStrideArray getBatchArray() {
        return data;
    }

    /**
     * The number of matrices in the batch.
     *
     * @return The number of matrices in the batch.
     */
    public int getBatchSize() {
        return data.batchSize;
    }

    /**
     * The stride size.
     *
     * @return The stride size.
     */
    public int getStrideSize() {
        return data.stride;
    }

    /**
     * Returns a matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     *
     * @param startRow The row the submatrices start on.
     * @param endRowExclsve The row the submatrices end on, exclusive.
     * @return A matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     */
    public MatricesStride subMatrixRows(int startRow, int endRowExclsve) {
        return new MatricesStride(handle, data, 1, width, colDist, 1, height);
    }

    /**
     * Returns a matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     *
     * @param startCol The col the submatrices start on.
     * @param endColExclsve The row the submatrices end on, exclusive.
     * @return A matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     */
    public MatricesStride subMatrixCols(int startCol, int endColExclsve) {
        return new MatricesStride(handle, data, height, 1, colDist, colDist, width);
    }

    /**
     * A sub batch of this batch.
     *
     * @param start The index of the first subMatrix.
     * @param length One after the index of the last submatrix.
     * @return A subbatch.
     */
    public MatricesStride subBatch(int start, int length) {
        return copyDimensions(data.subBatch(start, length));

    }

    /**
     * If this matrices stride were to be represented as a single matrix, then
     * this would be its width.
     *
     * @return The width of a suggested representing all inclusive matrix.
     */
    private int totalWidth() {
        return data.stride * data.batchSize + width;
    }

    /**
     * Fills all elements of these matrices with the given values. This method
     * alocated two gpu arrays, so it probably should not be called for many
     * batches.
     *
     * @param scalar To fill these matrices.
     * @return This.
     */
    public MatricesStride fill(double scalar) {
        if (data.stride <= colDist * width) {
            if (colDist == height) data.fill(handle, scalar, 1);
            else if (height == 1) data.fill(handle, scalar, colDist);
            else data.fillMatrix(handle, height, totalWidth(), colDist, scalar);
        } else {
            try (DArray workSpace = DArray.empty(width * width)) {

                MatricesStride empty = new Matrix(handle, workSpace, height, width).repeating(data.batchSize);

                add(false, scalar, empty, 0, workSpace);
            }
        }
        return this;
    }

    /**
     * The underlying data.
     *
     * @return The underlying data.
     */
    public DArray dArray() {
        return data;
    }

    /**
     * The handle used for this's operations.
     *
     * @return The handle used for this's operations.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * Adds matrices to these matrices.
     *
     * @param transpose Should toAdd be transposed.
     * @param timesToAdd Multiply toAdd before adding.
     * @param toAdd The matrices to add to these.
     * @param timesThis multiply this before adding.
     * @param workSpace workspace should be width^2 length.
     * @return
     */
    public MatricesStride add(boolean transpose, double timesToAdd, MatricesStride toAdd, double timesThis, DArray workSpace) {

        Matrix id = Matrix.identity(handle, width, workSpace);

        addProduct(transpose, false, timesToAdd, toAdd, id.repeating(data.batchSize), timesThis);
        return this;
    }

    /**
     * Row operation that convert this matrix into a diagonal matrix.
     *
     * @return These matrices.
     */
    public MatricesStride diagnolize(DArray pivot) {
        throw new UnsupportedOperationException("This method is not yet written.");
    }

    @Override
    public int getColDist() {
        return colDist;
    }

}





////Former eVec method.
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec() After eigen subtraction:\n" + toString());
//        getPointers().LUFactor(handle, pivot, info);
//
////        try(DArray ld = DArray.empty(4); DArray ud = DArray.empty(4)){
////            //delte this hole section.  Its just for debugging.
////            Matrix l = getSubMatrix(0).lowerLeftUnitDiagonal(ld);
////            Matrix u = getSubMatrix(0).upperRight(ud);
////            System.out.println("checking LU product: \n" + l.multiplyAndSet(l, u));
////        }
////        
////        
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec() pivot\n" + pivot.toString());
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec() after LU:\n" + toString());
//
//        JCudaWrapper.algebra.Vector[][] m = parition();
//
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec()  m = " + Arrays.deepToString(m));
//        eVector.get(height - 2)
//                .set(m[width - 2][height - 1])
//                .ebeDivide(m[width - 2][height - 2])
//                .multiply(-1);
//
//        if (eVector.getSubVecDim() == 3)
//            eVector.getElement(0)
//                    .set(eVector.getElement(1))
//                    .multiply(-1, m[0][1])
//                    .add(-1, m[0][2])
//                    .ebeDivide(m[0][0]);
//
////        System.out.println("Before pivoting: " + eVector);
////        KernelManager.get("unPivotVec").map(//TODO: understand why unpivoting seems to give the wrong answer, and not unpivoting seems to get it right.
////                        handle, 
////                        pivot, 
////                        height, 
////                        eVector.dArray(), 
////                        eVector.inc(), 
////                        data.batchCount(),
////                        IArray.cpuPointer(data.stride)
////                );
////        System.out.println("After pivoting:  " + eVector);
------------------------------------------------
package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.resourceManagement.Handle;
import java.awt.Dimension;
import java.util.Arrays;
import org.apache.commons.math3.exception.*;
import org.apache.commons.math3.linear.*;
import JCudaWrapper.array.DSingleton;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;

/**
 * Represents a matrix stored on the GPU. For more information on jcuda
 *
 * TODO: implement extensions of this class to include banded, symmetric banded,
 * triangular packed, symmetric matrices.
 *
 * http://www.jcuda.org/jcuda/jcublas/JCublas.html
 */
public class Matrix implements AutoCloseable, ColumnMajor {

    /**
     * The number of rows in the matrix.
     */
    private final int height;

    /**
     * The number of columns in the matrix.
     */
    private final int width;

    /**
     * The distance between the first element of each column in memory.
     * <p>
     * Typically, this is equal to the matrix height, but if this matrix is a
     * submatrix, `colDist` may differ, indicating that the matrix data is
     * stored with non-contiguous elements in memory.
     * </p>
     */
    public final int colDist;

    /**
     * The underlying GPU data storage for this matrix.
     */
    protected final DArray data;

    /**
     * Handle for managing JCublas operations, usually unique per thread.
     */
    protected Handle handle;

    /**
     * Constructs a new Matrix from a 2D array, where each inner array
     * represents a column of the matrix.
     *
     * @param handle The handle for JCublas operations, required for matrix
     * operations on the GPU.
     * @param matrix A 2D array, where each sub-array is a column of the matrix.
     */
    public Matrix(Handle handle, double[][] matrix) {
        this(handle, matrix[0].length, matrix.length);
        Matrix.this.set(0, 0, matrix);
    }

    /**
     * Constructs a Matrix from a single array representing a column-major
     * matrix.
     *
     * @param array The array storing the matrix in column-major order.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(Handle handle, DArray array, int height, int width) {
        this(handle, array, height, width, height);
    }

    /**
     * Constructs a new Matrix from an existing RealMatrix object, copying its
     * data to GPU memory.
     *
     * @param mat The matrix to be copied to GPU memory.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(Handle handle, RealMatrix mat) {
        this(handle, mat.getData());
    }

    /**
     * Creates a shallow copy of an existing Matrix, referencing the same data
     * on the GPU without copying. Changes to this matrix will affect the
     * original and vice versa.
     *
     * @param mat The matrix to create a shallow copy of.
     */
    public Matrix(Matrix mat) {
        this(mat.handle, mat.data, mat.height, mat.width, mat.colDist);
    }

    /**
     * Constructs a new Matrix from an existing data pointer on the GPU.
     *
     * @param vector Pointer to the data on the GPU.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param distBetweenFirstElementOfColumns The distance between the first
     * element of each column in memory, usually equal to height. If this is a
     * submatrix, it may differ.
     * @param handle The handle for GPU operations.
     */
    public Matrix(Handle handle, DArray vector, int height, int width, int distBetweenFirstElementOfColumns) {
//        if (!GPU.IsAvailable())
//            throw new RuntimeException("GPU is not available.");

        this.height = height;
        this.width = width;
        this.data = vector;
        this.handle = handle;
        this.colDist = distBetweenFirstElementOfColumns;
    }

    /**
     * Constructs an empty matrix of specified height and width.
     *
     * @param handle The handle for GPU operations.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     */
    public Matrix(Handle handle, int height, int width) {
        this(handle, DArray.empty(height * width), height, width);
    }

    /**
     * Returns the height (number of rows) of the matrix.
     *
     * @return The number of rows in the matrix.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the width (number of columns) of the matrix.
     *
     * @return The number of columns in the matrix.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Multiplies two matrices, adding the result into this matrix. The result
     * is inserted into this matrix as a submatrix.
     *
     * @param transposeA True if the first matrix should be transposed.
     * @param transposeB True if the second matrix should be transposed.
     * @param timesAB Scalar multiplier for the product of the two matrices.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param timesThis Scalar multiplier for the elements in this matrix.
     * @return This matrix after the operation.
     */
    public Matrix addProduct(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {

        Dimension aDim = new Dimension(transposeA ? a.height : a.width, transposeA ? a.width : a.height);
        Dimension bDim = new Dimension(transposeB ? b.height : b.width, transposeB ? b.width : b.height);
        Dimension result = new Dimension(bDim.width, aDim.height);

        checkRowCol(result.height - 1, result.width - 1);

        data.addProduct(handle, transposeA, transposeB,
                aDim.height, bDim.width, aDim.width, timesAB,
                a.data, a.colDist, b.data, b.colDist,
                timesThis, colDist);
        return this;
    }

    /**
     * @see Matrix#addProduct(processSupport.Handle, boolean, boolean, double,
     * algebra.Matrix, algebra.Matrix, double) timesThis is set to 0, transpose
     * values are false, and timesAB is 1.
     * @param a To be multiplied by the first matrix.
     * @param b To be multiplied by the second matrix.
     * @return This matrix.
     */
    public Matrix setToProduct(Matrix a, Matrix b) {
        return addProduct(false, false, 1, a, b, 0);
    }

    /**
     * @see Matrix#addProduct(processSupport.Handle, boolean, boolean, double,
     * algebra.Matrix, algebra.Matrix, double) Uses the default handle.
     * @param transposeA True to transpose A, false otherwise.
     * @param transposeB True to transpose B, false otherwise.
     * @param timesAB TO me multiplied by AB.
     * @param a The A matrix.
     * @param b The B matrix.
     * @param timesThis To be multiplied by this.
     * @return this.
     */
    public Matrix setToProduct(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {
        return addProduct(transposeA, transposeB, timesAB, a, b, timesThis);
    }

    /**
     * Returns the row index corresponding to the given column-major vector
     * index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The row index.
     */
    private int rowIndex(int vectorIndex) {
        return vectorIndex % height;
    }

    /**
     * Returns the column index corresponding to the given column-major vector
     * index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The column index.
     */
    private int columnIndex(int vectorIndex) {
        return vectorIndex / height;
    }

    /**
     * Copies a matrix to the GPU and stores it in the internal data structure.
     *
     * @param toRow The starting row index in this matrix.
     * @param toCol The starting column index in this matrix.
     * @param matrix The matrix to be copied, represented as an array of
     * columns.
     */
    private final void set(int toRow, int toCol, double[][] matrix) {
        for (int col = 0; col < Math.min(width, matrix.length); col++) {
            data.set(handle, matrix[col], index(toRow, toCol + col));
        }
    }

    /**
     * Performs matrix addition or subtraction.
     *
     * <p>
     * This function computes this = alpha * A + beta * B, where A and B are
     * matrices.
     * </p>
     *
     * @param handle The handle.
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
    public Matrix setSum(Handle handle, boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {

        if (transA) {
            checkRowCol(a.width - 1, a.height - 1);
        } else {
            checkRowCol(a.height - 1, a.width - 1);
        }
        if (transB) {
            checkRowCol(b.width - 1, b.height - 1);
        } else {
            checkRowCol(b.height - 1, b.width - 1);
        }

        data.setSum(handle, transA, transB, height, width, alpha, a.data, a.colDist, beta, b.data, b.colDist, colDist);

        return this;
    }

    /**
     * @see Matrix#setSum(boolean, boolean, double, algebra.Matrix, double, algebra.Matrix) Uses default handle.
     * @param transA True to transpose A.
     * @param transB True to transpose B.
     * @param alpha the multiply by A.
     * @param a The A matrix.
     * @param beta To multiply by B.
     * @param b The B matrix.
     * @return This.
     */
    public Matrix setSum(boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {
        return Matrix.this.setSum(handle, transA, transB, alpha, a, beta, b);
    }

    /**
     * @see Matrix#setSum(boolean, boolean, double, algebra.Matrix, double, algebra.Matrix) Uses default handle.
     * @param alpha the multiply by A.
     * @param a The A matrix.
     * @param beta To multiply by B.
     * @param b The B matrix.
     * @return This.
     */
    public Matrix setSum(double alpha, Matrix a, double beta, Matrix b) {
        return Matrix.this.setSum(handle, false, false, alpha, a, beta, b);
    }

    /**
     * Multiplies everything in this matrix by a scalar
     *
     * @param d The scalar that does the multiplying.
     * @return A new matrix equal to this matrix times a scalar.
     */
    public Matrix multiply(double d) {
        return Matrix.this.setSum(d, this, 0, this);
    }

    /**
     * Fills this matrix with @code{d}, overwriting whatever is there.
     *
     * @param scalar The value to fill the matrix with.
     * @return this.
     */
    public Matrix fill(double scalar) {
        data.fillMatrix(handle, height, width, colDist, scalar);
        return this;
    }

    /**
     * Adds a scalar to every element of this matrix.
     *
     * @param d The scalar to be added.
     * @return this.
     */
    public Matrix add(double d) {
        try (DSingleton sing = new DSingleton(handle, d)) {
            KernelManager.get("addScalarToMatrix").map(handle, sing, colDist, data, height, size());
            return this;
        }
    }

    /**
     * Inserts anther matrix into this matrix at the given index.
     *
     * @param handle The handle with which to perform this operation.
     * @param other The matrix to be inserted
     * @param row the row in this matrix where the first row of the other matrix
     * is inserted.
     * @param col The column in this matrix where the first row of the other
     * matrix is inserted.
     * @return this.
     *
     */
    public Matrix insert(Handle handle, Matrix other, int row, int col) {
        checkSubMatrixParameters(row, row + other.height, col, col + other.width);

        getSubMatrix(row, row + other.height, col, col + other.width)
                .setSum(1, other, 0, other);

        return this;
    }

    /**
     * @see Matrix#insert(processSupport.Handle, algebra.Matrix, int, int)
     * except with default handle.
     *
     * @param other
     * @param row
     * @param col
     * @return
     */
    public Matrix insert(Matrix other, int row, int col) {
        return insert(handle, other, row, col);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return rows().toString();

    }

    /**
     * Gets the entry at the given row and column.
     *
     * @param row The row of the desired entry.
     * @param column The column of the desired entry.
     * @return The entry at the given row and column.
     */
    public double get(int row, int column) {
        return data.get(index(row, column)).getVal(handle);

    }

    /**
     * The dimensions of a submatrix.
     *
     * @param startRow The top row of the submatrix.
     * @param endRow The bottom row of the submatrix, exclusive.
     * @param startColumn The first column of a submatrix.
     * @param endColumn The last column of the submatrix, exclusive.
     * @return The dimensions of a submatrix.
     */
    private Dimension subMatrixDimensions(int startRow, int endRow, int startColumn, int endColumn) {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);
        return new Dimension(endColumn - startColumn, endRow - startRow);
    }

    /**
     * Does some basic checks on the validity of the subMatrix parameters. Throw
     * exceptions if there are any problems.
     *
     * @param startRow inclusive
     * @param endRow exclusive
     * @param startColumn inclusive
     * @param endColumn exclusive
     * @throws OutOfRangeException
     * @throws NumberIsTooSmallException
     */
    private void checkSubMatrixParameters(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkRowCol(endRow - 1, endColumn - 1);
        checkRowCol(startRow, startColumn);
        if (startColumn > endColumn) {
            throw new NumberIsTooSmallException(endColumn, startColumn, true);
        }
        if (startRow > endRow) {
            throw new NumberIsTooSmallException(endRow, startRow, true);
        }

    }

    /**
     * Passes by reference. Changes to the sub matrix will effect the original
     * matrix and vice versa.
     *
     * @param startRow The starting row of the submatrix.
     * @param endRow The end row of the submatrix exclusive.
     * @param startColumn The starting column of the submatrix.
     * @param endColumn The end column exclusive.
     * @return The submatrix.
     * @throws OutOfRangeException
     * @throws NumberIsTooSmallException
     */
    public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        return new Matrix(
                handle,
                data.subArray(index(startRow, startColumn), (dim.width - 1) * colDist + dim.height),
                dim.height,
                dim.width,
                colDist);
    }

    /**
     * A submatrix consisting of the given rows.
     *
     * @param startRow The first row.
     * @param endRow The last row, exclusive.
     * @return A submatrix from the given rows.
     */
    public Matrix getRows(int startRow, int endRow) {
        return getSubMatrix(startRow, endRow, 0, getWidth());
    }

    /**
     * A submatrix consisting of the given columns.
     *
     * @param startCol The first column.
     * @param endCol The last column, exclusive.
     * @return A submatrix from the given rows.
     */
    public Matrix getColumns(int startCol, int endCol) {
        return getSubMatrix(0, getHeight(), startCol, endCol);
    }

    /**
     * If the row is outside of this matrix, an exception is thrown.
     *
     * @param row The row to be checked.
     * @throws OutOfRangeException
     */
    private void checkRow(int row) throws OutOfRangeException {
        if (row < 0 || row >= height) {
            throw new OutOfRangeException(row, 0, height);
        }
    }

    /**
     * If the column is outside of this matrix, an exception is thrown.
     *
     * @param col The column to be checked.
     * @throws OutOfRangeException
     */
    private void checkCol(int col) {
        if (col < 0 || col >= width) {
            throw new OutOfRangeException(col, 0, width);
        }
    }

    /**
     * If either the row or column are out of range, an exception is thrown.
     *
     * @param row The row to be checked.
     * @param col The column to be checked.
     */
    private void checkRowCol(int row, int col) throws OutOfRangeException {
        checkRow(row);
        checkCol(col);
    }

    /**
     * Checks if any of the objects passed are null, and if they are, throws a
     * null argument exception.
     *
     * @param o To be checked for null values.
     */
    private void checkForNull(Object... o) {
        if (Arrays.stream(o).anyMatch(obj -> obj == null)) {
            throw new NullArgumentException();
        }
    }

    /**
     * The number of elements in this matrix.
     *
     * @return The number of elements in this matrix.
     */
    public int size() {
        return width * height;
    }

    /**
     * Checks if the two methods are equal to within an epsilon margin of error.
     *
     * @param other A matrix that might be equal to this one.
     * @param epsilon The acceptable margin of error.
     * @param workSpace Should be the size of the matrix.
     * @return True if the matrices are very close to one another, false
     * otherwise.
     */
    public boolean equals(Matrix other, double epsilon, DArray workSpace) {
        if (height != other.height || width != other.width) return false;

        return new Matrix(handle, data, height, width).setSum(1, this, -1, other)
                .frobeniusNorm(workSpace.subArray(0, width)) <= epsilon;
    }

    /**
     * A copy of this matrix.
     *
     * @return A copy of this matrix.
     */
    public Matrix copy() {
        if (height == colDist) {
            return new Matrix(handle, data.copy(handle), height, width);
        }

        Matrix copy = new Matrix(handle, height, width);

        copy.setSum(1, this, 0, this);

        return copy;
    }

    /**
     * Sets an entry.
     *
     * @param row The row of the entry.
     * @param column The column of the entry.
     * @param value The value to be placed at the entry.
     */
    public void set(int row, int column, double value) {
        data.set(handle, index(row, column), value);
    }

    /**
     * gets the row.
     *
     * @param row the index of the desired row.
     * @return The row at the requested index.
     * @throws OutOfRangeException
     */
    public Vector getRow(int row) throws OutOfRangeException {
        return new Vector(handle, data.subArray(row), colDist);
    }

    /**
     * Gets the requested column.
     *
     * @param column The index of the desired column.
     * @return The column at the submited index.
     * @throws OutOfRangeException
     */
    public Vector getColumn(int column) throws OutOfRangeException {
        return new Vector(handle, data.subArray(index(0, column), height), 1);
    }

    /**
     * A copy of this matrix as a 2d cpu array. TDOD: by iterating along columns
     * and then transposing this method can be made faster.
     *
     * @return A copy of this matrix as a 2d cpu array.
     */
    public double[][] get() {
        double[][] getData = new double[height][];
        Arrays.setAll(getData, i -> getRow(i).vecGet());
        return getData;
    }

    /**
     * The trace of this matrix.
     */
    public double getTrace() throws NonSquareMatrixException {
        if (height != width) throw new NonSquareMatrixException(width, height);
        return data.dot(handle, new DSingleton(handle, 1), 0, width + 1);
    }

    /**
     * A hash code for this matrix. This is computed by importing the entire
     * matrix into cpu memory. TODO: do better.
     *
     * @return a hash code for this matrix.
     */
    @Override
    public int hashCode() {
        return new Array2DRowRealMatrix(get()).hashCode();
    }

    /**
     * Created a matrix from a double[] representing a column vector.
     *
     * @param vec The column vector.
     * @param handle
     * @return A matrix representing a column vector.
     */
    public static Matrix fromColVec(double[] vec, Handle handle) {
        Matrix mat = new Matrix(handle, vec.length, 1);
        mat.data.set(handle, vec);
        return mat;
    }

    /**
     * The identity Matrix.
     *
     * @param n the height and width of the matrix.
     * @param hand
     * @param holdIdentity The underlying array that will hold the identity
     * matrix.
     * @return The identity matrix.
     */
    public static Matrix identity(Handle hand, int n, DArray holdIdentity) {

        Matrix ident = new Matrix(hand, holdIdentity, n, n);
        ident.data.fill0(hand);

        ident.data.add(hand, 1, DSingleton.oneOne, 0, n + 1);

        return ident;
    }

    /**
     * The identity Matrix.
     *
     * @param n the height and width of the matrix.
     * @param hand
     * @return The identity matrix.
     */
    public static Matrix identity(Handle hand, int n) {
        return identity(hand, n, DArray.empty(n * n));
    }

    /**
     * Raise this square matrix to a power. This method may use a lot of
     * auxiliary space and allocate new memory multiple times. TODO: fix this.
     *
     * @param p The power.
     * @return This matrix raised to the given power.
     * @throws NotPositiveException
     * @throws NonSquareMatrixException
     */
    public Matrix power(int p) throws NotPositiveException, NonSquareMatrixException {

        if (p < 0) throw new NotPositiveException(p);
        if (height != width) throw new NonSquareMatrixException(width, height);
        if (p == 0) return identity(handle, width);

        if (p % 2 == 0) {
            power(p / 2);
            return Matrix.this.setToProduct(this, this);
        } else {
            try (Matrix copy = copy()) {
                return Matrix.this.setToProduct(copy, power(p - 1));
            }
        }
    }

    /**
     * @see Matrix#setColumnVector(int,
     * org.apache.commons.math3.linear.RealVector)
     *
     * @param column The index of the column to be set.
     * @param vector The vector to be put in the desired location.
     * @throws OutOfRangeException
     * @throws MatrixDimensionMismatchException
     */
    public void setColumnVector(int column, Vector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        data.set(handle, vector.dArray(), index(0, column), 0, 0, vector.colDist, Math.min(height, vector.dim()));
    }

    /**
     * @see Matrix#setRowVector(int, org.apache.commons.math3.linear.RealVector)
     */
    public void setRowVector(int row, Vector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        data.set(handle, vector.dArray(), index(row, 0), 0, colDist, vector.colDist, Math.min(width, vector.dim()));
    }

    /**
     * transposes this matrix.
     *
     * @return
     */
    public Matrix transposeMe() {
        return setSum(true, false, 1, this, 0, this);
    }

    /**
     * A vector containing the dot product of each column and itself.
     *
     * @param workspace Should be as long as the width.
     * @return A vector containing the dot product of each column and itself.
     */
    public Vector columnsSquared(DArray workspace) {

        Vector cs = new Vector(handle, data, 1);

        VectorsStride columns = columns();

        cs.addBatchVecVecMult(1, columns, columns, 0);

        return cs;
    }

    /**
     * The norm of this vector.
     *
     * @param workspace Needs to be width long
     * @return
     */
    public double frobeniusNorm(DArray workspace) {
        return Math.sqrt(columnsSquared(workspace).getL1Norm());

    }

    /**
     * There should be one handle per thread.
     *
     * @param handle The handle used by this matrix.
     */
    public void setHandle(Handle handle) {
        this.handle = handle;
    }

    /**
     * There should be one handle per thread.
     *
     * @return The handle used by this matrix.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * Closes the underlying data of this method.
     */
    @Override
    public void close() {
//        if (colDist != height) {
//            throw new IllegalAccessError("You are cleaning data from a sub Matrix");
//        }
        data.close();
    }

    /**
     * Adds the outer product of a and b to this matrix .
     *
     * @param a A column.
     * @param b A row
     * @return The outer product of a and b.
     */
    public Matrix addOuterProduct(Vector a, Vector b) {

        data.outerProd(handle, height, width, 1, a.dArray(), a.colDist, b.dArray(), b.colDist, colDist);

        return this;
    }

    /**
     * The underlying column major data.
     *
     * @return The underlying column major data.
     */
    public DArray dArray() {
        return data;
    }

    /**
     * Creates a matrix from the underlying data in this matrix with a new
     * height, width, and distance between columns. Note that if the distance
     * between columns in the new matrix is less that in this matrix, the matrix
     * will contain data that this one does not.
     *
     * @param newHieght The height of the new matrix.
     * @param newWidth The width of the new matrix.
     * @param newColDist The distance between columns of the new matrix. By
     * setting the new column distance to be less than or greater than the old
     * one, the new matrix may have more or fewer elements.
     * @return A shallow copy of this matrix that has a different shape.
     *
     */
    public Matrix newDimensions(int newHieght, int newWidth, int newColDist) {
        return new Matrix(handle, data, newHieght, newWidth, newColDist);
    }

    /**
     * Creates a matrix from the underlying data in this matrix with a new
     * height, width, and distance between columns. Note that if the distance
     * between columns in the new matrix is less that in this matrix, the matrix
     * will contain data that this one does not.
     *
     * @param newHieght The height of the new matrix. The width*colDist should
     * be divisible by this number.
     * @return A shallow copy of this matrix that has a different shape.
     *
     */
    public Matrix newDimensions(int newHieght) {
        return newDimensions(newHieght, size() / newHieght, newHieght);
    }

    /**
     * The distance between the 1st element of each column in column major
     * order.
     *
     * @return The distance between the first element of each column in column
     * major order.
     */
    @Override
    public int getColDist() {
        return colDist;
    }

    /**
     * The columns of this matrix.
     *
     * @return The columns of this matrix.
     */
    public VectorsStride columns() {
        return new VectorsStride(handle, data, 1, height, colDist, width);
    }

    /**
     * The rows of this matrix.
     *
     * @return The columns of this matrix.
     */
    public VectorsStride rows() {
        return new VectorsStride(handle, data, colDist, width, 1, height);
    }

    /**
     * A single array containing a copy of the data in this matrix in column
     * major order.
     *
     * @return A single array containing a copy of the data in this matrix in
     * column major order.
     */
    public double[] colMajor() {
        return dArray().get(handle);
    }

    /**
     * This matrix repeating itself in a batch.
     *
     * @param batchSize The size of the batch.
     * @return This matrix repeating itself in a batch.
     */
    public MatricesStride repeating(int batchSize) {
        return new MatricesStride(handle, data, height, width, colDist, 0, batchSize);
    }

    /**
     * This method extracts the lower left corner from this matrix. 1's are
     * place on the diagonal. This is only meant for square matrices that have
     * undergone LU factorization.
     *
     * @param putLHere Where the new matrix is to be placed.
     * @return
     */
    public Matrix lowerLeftUnitDiagonal(DArray putLHere) {
        Matrix l = new Matrix(handle, putLHere, height, width).fill(0);
        for (int i = 0; i < height - 1; i++) {
            l.getColumn(i).getSubVector(i + 1, height - i - 1)
                    .set(getColumn(i).getSubVector(i + 1, height - i - 1));
            l.set(i, i, 1);
        }
        l.set(height - 1, height - 1, 1);
        return l;
    }

    /**
     * This method extracts the upper right corner from this matrix. This is
     * only meant for square matrices that have undergone LU factorization.
     *
     * @param putUHere Where the new matrix is to be placed.
     * @return
     */
    public Matrix upperRight(DArray putUHere) {
        Matrix u = new Matrix(handle, putUHere, height, width).fill(0);
        for (int i = 0; i < height; i++)
            u.getColumn(i).getSubVector(0, i + 1)
                    .set(getColumn(i).getSubVector(0, i + 1));

        return u;
    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                DArray a = new DArray(hand, -1, 2, 3, 2, 4, 5, 3, 5, 6);
                DArray l = DArray.empty(9); DArray u = DArray.empty(9);
                IArray info = IArray.empty(1); IArray pivot = IArray.empty(3);) {

            Matrix m = new Matrix(hand, a, 3, 3);

            m.power(2);
            MatricesStride ms = m.repeating(1);

            System.out.println("m = \n" + m.toString() + "\n");

            Eigen eigen = new Eigen(ms);

            for (int i = 0; i < m.height; i++) {
                double eVal = eigen.values.get(i).get(0);
                Vector eVec = eigen.vectors.getMatrix(0).getColumn(i);

                System.out.println("\nEigen value " + i + ":\n " + eVal);
                System.out.println("Eigen vector " + i + ":\n " + eVec);

                System.out.println("m = \n" + m);

                System.out.println("Checking: is the vector = \n"
                        + eVec.addProduct(
                                false,
                                1 / eVal,
                                m,
                                eVec,
                                0
                        )
                );
            }
        }
    }

    /**
     * The number op non zeroe elements in this matrix.
     *
     * @return The number op non zeroe elements in this matrix.
     */
    public int numNonZeroes() {
        double[] columnMajor = colMajor();
        return (int) Arrays.stream(columnMajor).filter(d -> Math.abs(d) > 1e-10).count();
    }

}
------------------------------------------------
package main;

import JCudaWrapper.algebra.Matrix;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import java.awt.image.WritableRaster;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG implements AutoCloseable {

    private StructureTensorMatrix stm;
    private int height, width;
    private Handle handle;

    /**
     *
     * @param imagePath The location of the image.
     * @param neighborhoodSize The size of the edges of each neighborhood
     * square.
     * @throws java.io.IOException If there's trouble loading the image.
     */
    public NeighborhoodPIG(String imagePath, int neighborhoodSize) throws IOException {
        handle = new Handle();        

        Matrix imageMat = processImage(imagePath, handle);
        
        Gradient grad = new Gradient(imageMat, handle);
        imageMat.close();
        stm = new StructureTensorMatrix(grad.x(), grad.y(), neighborhoodSize);
        grad.close();

    }

    /**
     * Writes a heat map orientation picture to the given file.
     *
     * @param writeTo The new orientation image.
     */
    public void orientationColored(String writeTo) {

        try (IArray rgb = stm.getRGB()) {

            BufferedImage image = new BufferedImage(height, width, BufferedImage.TYPE_INT_RGB);
            WritableRaster raster = image.getRaster();

            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++)
                    raster.setPixel(col, row, rgb.get(handle, (col * height + row) * 3, 3));

            try {
                ImageIO.write(image, "png", new File(writeTo));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    public static void main(String[] args) throws IOException {
//        NeighborhoodPIG np = new NeighborhoodPIG("images/input/debug.jpeg", 1);

        NeighborhoodPIG np = new NeighborhoodPIG("images/input/test.jpeg", 1);

        np.orientationColored("images/output/test.png");

//        System.out.println(np.stm.setOrientations());
    }

    /**
     * Method to load a .tif image and convert it into a single-dimensional
     * array in column-major format.
     *
     * @param imagePath The path to the .tif image file
     * @param handle
     * @return A matrix of the image data.
     * @throws IOException if the image cannot be loaded
     */
    public final Matrix processImage(String imagePath, Handle handle) throws IOException {

        BufferedImage image = ImageIO.read(new File(imagePath));

        if (image.getType() != BufferedImage.TYPE_BYTE_GRAY)
            image = convertToGrayscale(image);

        //TODO: delete next two lines.
//        int d = 55;//55 seems to be the maximum
//        image = image.getSubimage(image.getWidth() / 2 - d / 2, image.getHeight() / 2 - d / 2, d, d);

        Raster raster = image.getRaster();

        width = image.getWidth();
        height = image.getHeight();

        double[] imageData = new double[width * height];

        Arrays.setAll(imageData, i -> raster.getSample(i / height, i % height, 0) / 255.0);

        Matrix mat = new Matrix(
                handle,
                new DArray(handle, imageData),
                height,
                width);

//        System.out.println(mat);
        return mat;
    }

    /**
     * Converts a given BufferedImage to grayscale.
     *
     * @param image The original BufferedImage
     * @return A grayscale BufferedImage
     */
    private BufferedImage convertToGrayscale(BufferedImage image) {

        BufferedImage grayImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D g2d = grayImage.createGraphics();
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();

        return grayImage; // Return the grayscale image
    }

    @Override
    public void close() {
        stm.close();
        handle.close();
    }
}
------------------------------------------------
package main;

import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.resourceManagement.Handle;

/**
 * This class implements element-by-element multiplication (EBEM) for
 * neighborhood-based matrix operations. It computes the sum of products from
 * neighborhoods in two input matrices, storing the results in the specified
 * vector.
 *
 * The input matrices are expected to have equal dimensions and column distances
 * (colDist).
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodProductSums implements AutoCloseable {

    private final Vector halfNOnes;
    private final Matrix ebeStorage, sumLocalRowElements;
    private final int nRad, height, width;

    /**
     * Constructs a {@code NeighborhoodProductSums} instance to compute the sum
     * of element-by-element products for neighborhoods within two matrices.
     *
     * @param handle A resource handle for creating internal matrices.
     * @param nRad Neighborhood radius; the distance from the center of a
     * neighborhood to its edge.
     * @param height The height of expected matrices. That is, matrices that
     * will be passed to the set method.
     * @param width The width of expected matrices.
     *
     */
    public NeighborhoodProductSums(Handle handle, int nRad, int height, int width) {
        this.nRad = nRad;
        this.height = height;
        this.width = width;
        sumLocalRowElements = new Matrix(handle, height, width).fill(0);//TODO:remove fill 0
        ebeStorage = new Matrix(handle, height, width);
        halfNOnes = new Vector(handle, nRad + 1).fill(1);
    }

    /**
     * Computes neighborhood element-wise multiplication of matrices a and b.
     * Divided into row and column stages for better performance. Then places in
     * result the summation of all the ebe products in the neighborhood of an
     * index pair in that index pair (column major order).
     *
     * @param a The first matrix.
     * @param b The second matrix.
     * @param result Store the result here in column major order. Note that the
     * increment of this vector is probably not one. 
     */
    public void set(Matrix a, Matrix b, Vector result) {

        Handle hand = a.getHandle();

        new Vector(hand, ebeStorage.dArray(), 1)
                .ebeSetProduct(
                        new Vector(hand, a.dArray(), 1),
                        new Vector(hand, b.dArray(), 1)
                );

        sumLocalRowElementsEdge();
        sumLocalRowElementsNearEdge();
        sumLocalRowElementsCenter();
        
        VectorsStride resultRows = result.subVectors(1, width, a.colDist, height);

        nSumEdge(resultRows);
        nSumNearEdge(resultRows);
        nSumCenter(resultRows);
        //TODO: pixels nearer the edges have lower sums.  They should probably be normalized for this.
    }

    /**
     * Handles column summation for the first and last columns (Stage I).
     *
     * @param inRowSum Matrix to store intermediate row sums.
     * @param ebeStorage Element-wise multiplied matrix.
     * @param nRad Neighborhood radius.
     * @param height Height of the matrix.
     * @param width Width of the matrix.
     * @param halfNOnes Vector of ones used for summing the first and last
     * columns.
     */
    private void sumLocalRowElementsEdge() {
        
        sumLocalRowElements.getColumn(0).setProduct(
                ebeStorage.getColumns(0, nRad + 1),
                halfNOnes
        );
        sumLocalRowElements.getColumn(width - 1).setProduct(
                ebeStorage.getColumns(width - nRad - 1, width),
                halfNOnes
        );        
    }

    /**
     * Handles column summation for columns near the first and last (Stage II).
     *
     * @param inRowSum Matrix to store intermediate row sums.
     * @param ebeStorage Element-wise multiplied matrix.
     * @param nRad Neighborhood radius.
     * @param width Width of the matrix.
     */
    private void sumLocalRowElementsNearEdge() {
        for (int i = 1; i < nRad + 1; i++) {
            sumLocalRowElements.getColumn(i).setSum(
                    1, ebeStorage.getColumn(i + nRad),
                    1, sumLocalRowElements.getColumn(i - 1)
            );
            int colInd = width - 1 - i;
            sumLocalRowElements.getColumn(colInd).setSum(
                    1, ebeStorage.getColumn(colInd - nRad),
                    1, sumLocalRowElements.getColumn(colInd + 1)
            );
        }
    }

    /**
     * Handles column summation for the central columns (Stage III).
     *
     * @param inRowSum Matrix to store intermediate row sums.
     * @param ebeStorage Element-wise multiplied matrix.
     * @param nRad Neighborhood radius.
     * @param width Width of the matrix.
     */
    private void sumLocalRowElementsCenter() {
        for (int colIndex = nRad + 1; colIndex + nRad < width; colIndex++) {
                       
            
            sumLocalRowElements.getColumn(colIndex).setSum(
                    -1, ebeStorage.getColumn(colIndex - nRad - 1),
                    1, ebeStorage.getColumn(colIndex + nRad)
            );
            sumLocalRowElements.getColumn(colIndex).add(1, sumLocalRowElements.getColumn(colIndex - 1));//todo: make this one command instead of two.

        }
    }

    /**
     * Handles row summation for the first and last rows (Stage I).
     *
     * @param nSums Matrix on the result vector to store the results.
     * @param inRowSum Matrix containing intermediate row sums.
     * @param nRad Neighborhood radius.
     * @param width Width of the matrix.
     * @param halfNOnes Vector of ones used for summing the first and last rows.
     */
    private void nSumEdge(VectorsStride resultRows) {
        resultRows.getVector(0).setProduct(
                halfNOnes,
                sumLocalRowElements.getRows(0, nRad + 1)
        );
        resultRows.getVector(height - 1).setProduct(
                halfNOnes,
                sumLocalRowElements.getRows(height - nRad - 1, height)
        );
    }

    /**
     * Handles row summation for rows near the first and last (Stage II).
     *
     * @param nSums Matrix on the result vector to store the results.
     * @param inRowSum Matrix containing intermediate row sums.
     * @param nRad Neighborhood radius.
     * @param height Height of the matrix.
     */
    private void nSumNearEdge(VectorsStride resultRows) {
        for (int i = 1; i < nRad + 1; i++) {

            int rowInd = i;
            resultRows.getVector(rowInd).setSum(
                    1, sumLocalRowElements.getRow(rowInd + nRad),
                    1, resultRows.getVector(rowInd - 1)
            );

            rowInd = height - 1 - i;
            resultRows.getVector(rowInd).setSum(
                    1, sumLocalRowElements.getRow(rowInd - nRad),
                    1, resultRows.getVector(rowInd + 1)
            );
        }
    }

    /**
     * Handles row summation for the central rows (Stage III).
     *
     * @param nSums Matrix on the result vector to store the results.
     * @param inRowSum Matrix containing intermediate row sums.
     * @param nRad Neighborhood radius.
     * @param height Height of the matrix.
     */
    private void nSumCenter(VectorsStride resultRows) {
        for (int rowIndex = nRad + 1; rowIndex < height - nRad; rowIndex++) {

            resultRows.getVector(rowIndex).setSum(
                    -1, sumLocalRowElements.getRow(rowIndex - nRad - 1),
                    1, sumLocalRowElements.getRow(rowIndex + nRad)
            ).add(1, resultRows.getVector(rowIndex - 1));
        }
    }

    /**
     * Cleans up allocated memory on the gpu.
     */
    @Override
    public void close() {
        halfNOnes.close();
        ebeStorage.close();
        sumLocalRowElements.close();
    }
}
------------------------------------------------
package main;

import JCudaWrapper.algebra.ColumnMajor;
import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
import JCudaWrapper.resourceManagement.Handle;
import java.awt.Color;
import java.util.Arrays;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrix implements AutoCloseable, ColumnMajor {

    /**
     * This matrix is a row of tensors. The height of this matrix is the height
     * of one tensor and the length of this matrix is the number of pixels times
     * the length of a tensor. The order of the tensors is column major, that
     * is, the first tensor corresponds to the pixel at column 0 row 0, the
     * second tensor corresponds to the pixel at column 0 row 1, etc...
     */
    private final MatricesStride strctTensors;

    private final Eigen eigen;
    private final Matrix orientation;
    private Handle handle;

    public StructureTensorMatrix(Matrix dX, Matrix dY, int neighborhoodRad) {

        handle = dX.getHandle();

        int height = dX.getHeight(), width = dX.getWidth();

        strctTensors = new MatricesStride(handle, 2, dX.size());//reset to 3x3 for dZ.

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(dX.getHandle(), neighborhoodRad, height, width)) {
            nps.set(dX, dX, strctTensors.get(0, 0));
            nps.set(dX, dY, strctTensors.get(0, 1));//TODO: Check other arrays for using length as the number of times something is done when increment says it should not be!
            nps.set(dY, dY, strctTensors.get(1, 1));

//            nps.set(dX, dZ, strctTensors.get(0, 2));
//            nps.set(dY, dZ, strctTensors.get(1, 2));
//            nps.set(dZ, dZ, strctTensors.get(2, 2));//Add these when working with dZ.
        }

        strctTensors.get(1, 0).set(strctTensors.get(0, 1));
//        strctTensors.get(2, 0).set(strctTensors.get(0, 2)); //engage for 3x3.
//        strctTensors.get(2, 1).set(strctTensors.get(1, 2));

        eigen = new Eigen(strctTensors);

        orientation = new Matrix(handle, height, width);
    }

    /**
     * The tensors are stored in one long row of 2x2 tensors in column major
     * order.
     *
     * @param picRow The row of the pixel in the picture for which the tensor's
     * index is desired.
     * @param picCol The row of the column in the picture for which the tensor's
     * image is desired.
     * @return The index of the beginning of the tensor matrix for the requested
     * pixel.
     */
    private int tensorFirstColIndex(int picRow, int picCol) {

        int tensorSize = strctTensors.height * strctTensors.height;

        return (picCol * orientation.getHeight() + picRow) * tensorSize;
    }

    /**
     * Gets the structure tensor from pixel at the given row and column of the
     * picture.
     *
     * @param row The row of the desired pixel.
     * @param col The column of the desired pixel.
     * @return The structure tensor for the given row and column.
     */
    public Matrix getTensor(int row, int col) {

        return strctTensors.getMatrix(index(row, col));
    }

    /**
     * All the eigen vectors with y less than 0 are mulitplied by -1.
     *
     * @return The eigenvectors.
     */
    public MatricesStride setVecs0ToPi() {
        MatricesStride eVecs = eigen.vectors;
        KernelManager.get("vecToNematic").mapToSelf(handle,
                eVecs.dArray(), eVecs.colDist,
                eVecs.getBatchSize() * eVecs.width
        );
        return eVecs;
    }

    /**
     * Sets the orientations from the eigenvectors.
     *
     * @return The orientation matrix.
     */
    public Matrix setOrientations() {
        KernelManager.get("atan2").map(handle,
                eigen.vectors.dArray(), eigen.vectors.getStrideSize(),
                orientation.dArray(), 1,
                orientation.size()
        );
        return orientation;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public Matrix getOrientations() {
        return orientation;
    }

    /**
     * Takes in a vector, in column major order for a matrix with orientation
     * dimensions, and returns a double[][] representing the vector in
     * [row][column] format.
     *
     * @param columnMajor A vector that is column major order of a matrix with
     * height orientation.height.
     * @param workSpace An auxillery workspace. It should be height in length.
     * @return a cpu matrix.
     */
    private double[][] getRows(Vector columnMajor) {
        return columnMajor.subVectors(1, orientation.getWidth(), orientation.colDist, orientation.getHeight())
                .copyToCPURows();
    }

//        (R, G, B) = (256*cos(x), 256*cos(x + 120), 256*cos(x - 120))  <- this is for 360.  For 180 maybe:
//    (R, G, B) = (256*cos(x), 256*cos(x + 60), 256*cos(x + 120))
    /**
     * Three matrices for red green blue color values.
     *
     * @return a column major array of colors. The first 3 elements are the RGB
     * values for the first color, etc...
     */
    public IArray getRGB() {

        setVecs0ToPi();
        setOrientations().multiply(2);
        
        IArray colors = IArray.empty(orientation.size() * 3);

        KernelManager.get("color").map(handle, orientation.dArray(), 1, colors, 3, orientation.size());

        return colors;

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        strctTensors.close();
        eigen.close();
        orientation.close();
    }

    @Override
    public int getColDist() {
        return orientation.getHeight();
    }

}
------------------------------------------------
package JCudaWrapper.algebra;

import java.util.Arrays;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.exception.NotPositiveException;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DSingleton;

/**
 * The {@code Vector} class extends {@code RealVector} and represents a vector
 * stored on the GPU. It relies on the {@code DArray} class for data storage and
 * the {@code Handle} class for JCublas operations.
 *
 * Vectors are horizontal matrices.
 */
public class Vector extends Matrix {

    /**
     * Constructs a new {@code Vector} from an existing data pointer on the GPU.
     * Do not use this constructor for vectors with inc == 0.
     *
     * @param data The {@code DArray} storing the vector on the GPU.
     * @param inc The increment between elements of the data that make of this
     * vector. If 0 is passed, methods that usde JCUDA matrix operations will
     * not work, while methods that use JCuda vector operations will. TODO: make
     * this better.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, DArray data, int inc) {
        super(handle, data, 1, Math.ceilDiv(data.length, inc), inc);
    }

    /**
     * Constructs a new {@code Vector} from a 1D array.
     *
     * @param array The array storing the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, double... array) {
        this(handle, new DArray(handle, array), 1);
    }

    /**
     * Constructs a new empty {@code Vector} of specified length.
     *
     * @param length The length of the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, int length) {
        this(handle, DArray.empty(length), 1);
    }

    /**
     * Gets the element at the given index.
     *
     * @param index The index of the desired element.
     * @return The element at the given index.
     * @throws OutOfRangeException If the element is out of range.
     */
    public double get(int index) throws OutOfRangeException {
        return data.get(index * inc()).getVal(handle);
    }

    /**
     * Sets the element at the given index.
     *
     * @param index The index whose element is to be set.
     * @param value The value to be placed at index.
     * @throws OutOfRangeException
     */
    public void set(int index, double value) throws OutOfRangeException {
        data.set(handle, index * inc(), value);
    }

    /**
     * The dimension of the vector. The number of elements in it.
     *
     * @return The dimension of the vector. The number of elements in it.
     */
    public int dim() {
        return inc() == 0 ? 1 : Math.ceilDiv(data.length, inc());
    }

    /**
     * Adds another vector times a scalar to this vector, changing this vector.
     *
     * @param mult A scalar to be multiplied by @code{v} before adding it to
     * this vector.
     * @param v The vector to be added to this vector.
     * @return This vector.
     */
    public Vector add(double mult, Vector v) {
        data.add(handle, mult, v.data, v.inc(), inc());
        return this;
    }

    /**
     * Adds the scalar to every element in this vector.
     *
     * @param scalar To be added to every element in this vector.
     * @return this.
     */
    @Override
    public Vector add(double scalar) {
        data.add(handle, scalar, DSingleton.oneOne, 0, 1);
        return this;

    }

    /**
     * multiplies this array by the scalar.
     *
     * @param scalar to multiply this array.
     * @return this.
     */
    public Vector multiply(double scalar) {
        data.multiply(handle, scalar, inc());
        return this;
    }

    /**
     * Sets all the values in this vector to that of the scalar.
     *
     * @param scalar The new value to fill this vector.
     * @return This vector.
     */
    public Vector fill(double scalar) {
        if (scalar == 0 && inc() == 1) {
            data.fill0(handle);
        } else {
            data.fill(handle, scalar, inc());
        }
        return this;
    }

    /**
     * Computes the dot product of this vector with another vector.
     *
     * @param v The other vector to compute the dot product with.
     * @return The dot product of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public double dotProduct(Vector v) {

        return data.dot(handle, v.data, v.inc(), inc());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector copy() {
        if (inc() == 1) {
            return new Vector(handle, data.copy(handle), inc());
        }
        Vector copy = new Vector(handle, dim());
        copy.data.set(handle, data, 0, 0, 1, inc(), dim());
        return copy;
    }

    /**
     * Computes the element-wise product of this vector and another vector.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeSetProduct(Vector a, Vector b) {

        return ebeAddProduct(a, b, 0);

    }

    /**
     * multiplies this vector by the given scalar and vector (element by
     * element).
     *
     * @param scalar The scalar times this.
     * @param a The vector that will be ebe times this.
     * @return this.
     */
    public Vector multiply(double scalar, Vector a) {
        return addEbeProduct(scalar, a, this, 0);
    }

    /**
     * Computes the element-wise product of this vector and another vector, and
     * adds it to this vector.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @param timesThis Multiply this matrix before adding the product of a and
     * b to it.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeAddProduct(Vector a, Vector b, double timesThis) {
        return addEbeProduct(1, a, b, timesThis);
    }

    /**
     * Computes the element-wise product of this vector and another vector, and
     * adds it to this vector.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @param timesAB A scalar to multiply by a and b.
     * @param timesThis multiply this vector before adding the product of a and
     * b.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector addEbeProduct(double timesAB, Vector a, Vector b, double timesThis) {

        data.addProductSymBandMatVec(handle, true,
                dim(), 0,
                timesAB, 
                a.data, a.inc(),
                b.data, b.inc(),
                timesThis, inc()
        );

        return this;
    }


    /**
     * Element by element division. Like most methods, it changes this vector.
     *
     * @param denominator the denominator.
     * @return this
     */
    public Vector ebeDivide(Vector denominator) {

        try {
            data.solveTriangularBandedSystem(handle, true, false, false,
                    denominator.dim(), 0, denominator.data, denominator.inc(), inc());
        } catch (Exception e) {
            if (Arrays.stream(denominator.toArray()).anyMatch(i -> i == 0))
                throw new ArithmeticException("Division by 0.");
            else throw e;

        }

        return this;
    }

    /**
     * A sub vector of this one.
     *
     * @param begin The index of this vector to begin the subvector.
     * @param length The number of elements in the subvector.
     * @return A subvector of this vector.
     * @throws NotPositiveException
     * @throws OutOfRangeException
     */
    public Vector getSubVector(int begin, int length) throws NotPositiveException, OutOfRangeException {
        return getSubVector(begin, length, 1);
    }

    /**
     * Returns a sub vector of this one. The vector is a shallow copy/ copy by
     * reference, changes made to the new vector will affect this one and vice
     * versa.
     *
     * @param begin Where the vector begins.
     * @param length The length of the new vector.The number of elements in the
     * vector.
     * @param increment The stride step of the new vector. For example, if this
     * value is set to 2, then the new vector will contain every other element
     * of this vector.
     * @return A sub vector of this vector.
     * @throws NotPositiveException
     * @throws OutOfRangeException
     */
    public Vector getSubVector(int begin, int length, int increment) throws NotPositiveException, OutOfRangeException {
        return new Vector(
                handle,
                data.subArray(begin * inc(), inc() * increment * (length - 1) + 1),
                inc() * increment
        );
    }

    /**
     *
     * @see Vector#setSubVector(int, org.apache.commons.math3.linear.RealVector)
     * Sets a subvector of this vector starting at the specified index using the
     * elements of the provided {@link Vector}.
     *
     * This method modifies this vector by copying the elements of the given
     * {@link Vector} into this vector starting at the position defined by the
     * index, with the length of the subvector determined by the dimension of
     * the given vector.
     *
     * @param i The starting index where the subvector will be set, scaled by
     * the increment (inc).
     * @param rv The {@link Vector} whose elements will be copied into this
     * vector.
     * @throws OutOfRangeException If the index is out of range.
     */
    public void setSubVector(int i, Vector rv) throws OutOfRangeException {
        data.set(handle, rv.data, i * inc(), 0, inc(), rv.inc(), rv.dim());
    }

    /**
     * Sets a portion of this vector to the contents of the given matrix.
     * Specifically, the method inserts the columns of the matrix into this
     * vector starting at the specified index and offsets each column by the
     * height of the matrix.
     *
     * @param toIndex the starting index in this vector where the subvector
     * (matrix columns) will be inserted
     * @param m the matrix whose columns are used to set the subvector in this
     * vector
     *
     * @throws IndexOutOfBoundsException if the specified index or resulting
     * subvector extends beyond the vector's bounds
     *
     *///TODO: This method can be made faster with multi threading (multiple handles)
    public void setSubVector(int toIndex, Matrix m) {
        for (int mCol = 0; mCol < dim(); mCol++) {
            setSubVector(toIndex + mCol * m.getHeight(), m.getColumn(mCol));
        }
    }

    /**
     * Sets a portion of this vector to the contents of the given Vector.
     *
     *
     * @param v The vector to copy from.
     * @return this
     */
    public Vector set(Vector v) {
        setSubVector(0, v);
        return this;
    }

    /**
     * Computes the cosine of the angle between this vector and the argument.
     *
     * @see Vector#cosine(org.apache.commons.math3.linear.RealVector)
     * @param other Vector
     * @return the cosine of the angle between this vector and v.
     */
    public double cosine(Vector other) {

        return dotProduct(other) / norm() * other.norm();
    }

    /**
     * The distance between this vector and another vector.
     *
     * @param v The other vector.
     * @param workSpace Should be as long as these vectors.
     * @return The distance to v.
     * @throws DimensionMismatchException
     */
    public double getDistance(Vector v, Vector workSpace) throws DimensionMismatchException {
        workSpace.fill(0);
        workSpace.setSum(1, v, -1, this);
        return workSpace.norm();
    }

    /**
     * {@inheritDoc}
     * @param alpha
     * @param a
     * @param beta
     * @param b
     * @return 
     */
    @Override
    public Vector setSum(double alpha, Matrix a, double beta, Matrix b) {
        super.setSum(alpha, a, beta, b);
        return this;
    }

    
    
    /**
     * The L_1 norm.
     */
    public double getL1Norm() {
        return data.sumAbs(handle, dim(), inc());
    }

    /**
     * The L_infinity norm
     */
    public double getLInfNorm() {
        return get(data.argMaxAbs(handle, dim(), inc()));
    }

    /**
     * Finds the index of the minimum or maximum element of the vector. This
     * method creates its own workspace equal in size to this.
     *
     * @param isMax True to find the argMaximum, false for the argMin.
     * @return The argMin or argMax.
     */
    private int getMinMaxInd(boolean isMax) {
        int argMaxAbsVal = data.argMaxAbs(handle, dim(), inc());
        double maxAbsVal = get(argMaxAbsVal);
        if (maxAbsVal == 0) {
            return 0;
        }
        if (maxAbsVal > 0 && isMax) {
            return argMaxAbsVal;
        }
        if (maxAbsVal < 0 && !isMax) {
            return argMaxAbsVal;
        }

        try (Vector sameSign = copy().add(maxAbsVal)) {
            return sameSign.data.argMinAbs(handle, dim(), inc());
        }
    }

    /**
     * The index of the minimum element.
     */
    public int minIndex() {
        return getMinMaxInd(false);
    }

    /**
     * The minimum value.
     */
    public double getMinValue() {
        return get(minIndex());
    }

    /**
     * The maximum index.
     */
    public int maxIndex() {
        return getMinMaxInd(true);
    }

    /**
     * The maximum value.
     */
    public double maxValue() {
        return get(maxIndex());
    }

    /**
     * @param v The vector with which this one is creating an outer product.
     * @param placeOuterProduct should have at least v.dim() * dim() elements.
     * @return The outer product. A new matrix.
     * @see Vector#outerProduct(org.apache.commons.math3.linear.RealVector)
     */
    public Matrix outerProduct(Vector v, DArray placeOuterProduct) {
        placeOuterProduct.outerProd(handle, dim(), v.dim(), 1, data, inc(), v.data, v.inc(), dim());
        return new Matrix(handle, placeOuterProduct, dim(), v.dim()).fill(0);
    }

    /**
     * @see Vector#projection(org.apache.commons.math3.linear.RealVector)
     *
     * @param v project onto.
     * @return The projection.
     *
     */
    public Vector projection(Vector v) throws DimensionMismatchException, MathArithmeticException {
        double[] dots = new double[2];

        data.dot(handle, v.data, v.inc(), inc(), dots, 0);
        v.data.dot(handle, v.data, v.inc(), inc(), dots, 1);

        return v.multiply(dots[0] / dots[1]);
    }

    /**
     * The cpu array that is a copy of this gpu vector.
     *
     * @return the array in the cpu.
     */
    public double[] toArray() {
        double[] to = new double[dim()];
        data.get(handle, to, 0, 0, 1, inc(), dim());
        return to;
    }

    /**
     * Turn this vector into a unit vector.
     */
    public void unitize() throws MathArithmeticException {
        Vector.this.multiply(1 / norm());
    }

    /**
     * The data underlying this vector.
     *
     * @return The underlying data from this vector.
     */
    public DArray dArray() {
        return data;
    }

    /**
     * A matrix representing the data underlying this Vector. Note, depending on
     * inc and colDist, the new matrix may have more or fewere elements than
     * this vector.
     *
     * @param height The height of the new matrix.
     * @param width The width of the new matrix.
     * @param colDist The disance between the first element of each column.
     * @return
     */
    public Matrix asMatrix(int height, int width, int colDist) {
        return new Matrix(handle, data, height, width, colDist);
    }

    /**
     * A matrix representing the data underlying this Vector. Note, depending on
     * inc and colDist, the new matrix may have more or fewere elements than
     * this vector.
     *
     * @param height The height of the new matrix. It should be divisible by the
     * number of elements in the underlying data.
     * @return A matrix containing the elements in the underlying data of this
     * vector.
     */
    public Matrix asMatrix(int height) {
        return new Matrix(handle, data, height, data.length / height, height);
    }

    /**
     * The handle for this matrix.
     *
     * @return The handle for this matrix.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * Batch vector vector dot product. This vector is set as the dot product of
     * a and b.
     *
     * @param timesAB Multiply this by the product of a and b.
     * @param a The first vector. A sub vector of a matrix or greater vector.
     * @param b The second vector. A sub vector of a matrix or greater vector.
     * @param timesThis multiply this before adding to it.
     */
    public void addBatchVecVecMult(double timesAB, VectorsStride a, VectorsStride b, double timesThis) {

        data.getAsBatch(inc(), 1, dim()).addProduct(handle,
                false, true,
                1, a.getSubVecDim(), 1,
                timesAB,
                a.data, a.inc(),
                b.data, b.inc(),
                timesThis, inc()
        );
    }

    /**
     * Batch vector vector dot product. This vector is set as the dot product of
     * a and b.
     *
     * @param a The first vector. A sub vector of a matrix or greater vector.
     *
     * @param b The second vector. A sub vector of a matrix or greater vector.
     * @return this
     *
     */
    public Vector setBatchVecVecMult(VectorsStride a, VectorsStride b) {
        addBatchVecVecMult(1, a, b, 0);
        return this;
    }

    /**
     * Partitions this vector into a sets of incremental subsets.
     *
     * @param numParts The number of subsets.
     * @return An array of incremental subsets.
     */
    public Vector[] parition(int numParts) {
        Vector[] part = new Vector[numParts];
        Arrays.setAll(part, i -> getSubVector(i, dim() / numParts, numParts));
        return part;
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * @param transposeMat Should the matrix be transposed.
     * @param timesAB Gets multiplied by the product.
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @param timesCurrent Multiply this before adding the product.
     * @return The product is placed in this and this is returned.
     */
    public Vector addProduct(boolean transposeMat, double timesAB, Vector vec, Matrix mat, double timesCurrent) {
        data.addProduct(handle,
                false, transposeMat,
                1, transposeMat ? mat.getHeight() : mat.getWidth(), vec.dim(),
                timesAB,
                vec.data, vec.inc(),
                mat.data, mat.colDist,
                timesCurrent, inc()
        );
        return this;
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @return The product is placed in this and this is returned.
     */
    public Vector setProduct(Vector vec, Matrix mat) {
        return addProduct(false, 1, vec, mat, 0);
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * This method does not work if increment does not equal 1. Try using matrix
     * methods instead or work with a vector that has an increment of 1.
     *
     * @param transposeMat Should the matrix be transposed.
     * @param timesAB Gets multiplied by the product.
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @param timesCurrent Multiply this before adding the product.
     * @return The product is placed in this and this is returned.
     */
    public Vector addProduct(boolean transposeMat, double timesAB, Matrix mat, Vector vec, double timesCurrent) {

        data.addProduct(handle, transposeMat, 
                mat.getHeight(), mat.getWidth(),                 
                timesAB, mat.data, mat.getColDist(), 
                vec.dArray(), vec.inc(), 
                timesCurrent, inc()
        );
        
        return this;
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @return The product is placed in this and this is returned.
     */
    public Vector setProduct(Matrix mat, Vector vec) {
        return addProduct(false, 1, mat, vec, 0);
    }

    /**
     * The increment between elements of this vector. This is the column
     * distance of this matrix.
     *
     * @return The increment between elements of this vector.
     */
    public int inc() {
        return colDist;
    }

    /**
     * A set of vectors contained within this vector.
     *
     * @param stride The distance between the first elements of each vector.
     * @param batchSize The number of sub vectors.
     * @param subVectorDim The number of elements in each sub vector.
     * @param subVectorInc The increment of each sub vector over this vector.
     * @return The set of sub vectors.
     */
    public VectorsStride subVectors(int stride, int subVectorDim, int subVectorInc, int batchSize) {
        return new VectorsStride(
                handle,
                data,
                inc() * subVectorInc,
                subVectorDim,
                inc() * stride,
                batchSize
        );
    }

    @Override
    public Matrix addProduct(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {
        throw new UnsupportedOperationException("Use the addProduct methods that take vectors as parameters instead.");
    }


    
    /**
     * This vector as a double array.
     *
     * @return
     */
    public double[] vecGet() {
        return data.getIncremented(handle, inc());
    }

    /**
     * The L2norm or magnitude of this vector.
     *
     * @return The norm of this vector.
     */
    public double norm() {
        return data.norm(handle, dim(), colDist);
    }

    @Override
    public String toString() {
        return Arrays.toString(toArray());
    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                DArray array = new DArray(hand, 1, 2, 3, 4, 5, 6);
                DArray a2 = new DArray(hand, 2, 2, 2, 2, 2, 2)) {
            Vector v = new Vector(hand, array, 1);
            v.ebeDivide(new Vector(hand, a2, 1));
            System.out.println(v);
        }
    }

}
------------------------------------------------
package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.KernelManager;
import java.util.Arrays;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class VectorsStride extends MatricesStride implements AutoCloseable {

    /**
     * The constructor.
     *
     * @param handle
     * @param data The underlying data.
     * @param subArrayInc The increment of each subvector.
     * @param dim The number of elements in each subvector.
     * @param strideSize The distance between the first elements of each
     * subevector.
     * @param batchSize The number of subvectors.
     */
    public VectorsStride(Handle handle, DArray data, int subArrayInc, int dim, int strideSize, int batchSize) {
        super(handle, data, 1, dim, subArrayInc, strideSize, batchSize);
    }

    /**
     * The constructor.
     *
     * @param handle
     * @param strideSize The stride size.
     * @param batchSize The number of vectors in the batch.
     * @param subVecDim The number of elements in each subvector.
     * @param inc The increment of each subvector.
     */
    public VectorsStride(Handle handle, int strideSize, int batchSize, int subVecDim, int inc) {
        this(
                handle,
                DArray.empty(DStrideArray.totalDataLength(strideSize, inc * subVecDim, batchSize)),
                inc,
                subVecDim,
                strideSize,
                batchSize
        );
    }

    /**
     * The element at the ith index in every subVector.
     *
     * @param i The index of the desired element.
     * @return The element at the ith index in every subVector.
     */
    public Vector getElement(int i) {
        return new Vector(
                handle,
                data.subArray(i * colDist),
                data.stride * colDist);
    }

    /**
     * Gets the subvector at the desired index.
     *
     * @param i The index of the desired subvector.
     * @return The subvector at the desired index.
     */
    public Vector getVector(int i) {
        return new Vector(handle, data.getBatchArray(i), colDist);

    }

    /**
     * The dimension of the subarrays.
     *
     * @return The number of elements in each sub array.
     */
    public int getSubVecDim() {
        return Math.ceilDiv(data.subArrayLength, colDist);
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param transposeMats Should the matrices be transposed.
     * @param mats The matrices.
     * @param vecs The vectors.
     * @param timesAB A scalar to multiply the product by.
     * @param timesThis A scalar to multiply this by before the product is add
     * here.
     */
    public VectorsStride addProduct(boolean transposeMats, double timesAB, MatricesStride mats, VectorsStride vecs, double timesThis) {
        //TODO: this method doesn't work because the reciecing vector, this, has the wrong dimensions.
        return addProduct(!transposeMats, timesAB, vecs, mats, timesThis);
        
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param transposeMats Should the matrices be transposed.
     * @param mats The matrices.
     * @param vecs The vectors.
     * @param timesAB A scalar to multiply the product by.
     * @param timesThis A scalar to multiply this by before the product is add
     * here.
     * @return this.
     */
    public VectorsStride addProduct(boolean transposeMats, double timesAB, VectorsStride vecs, MatricesStride mats, double timesThis) {
        super.addProduct(false, transposeMats, timesAB, vecs, mats, timesThis);
        return this;
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     * @return this
     */
    public VectorsStride setProduct(VectorsStride vecs, MatricesStride mats) {
        return addProduct(false, 1, vecs, mats, 0);
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     * @return this
     */
    public VectorsStride setProduct(MatricesStride mats, VectorsStride vecs) {
        addProduct(false, 1, mats, vecs, 0);
        return this;
    }

    /**
     * Partitions this into an array of vectors so that v[i].get(j) is the ith
     * element in the jth vector.
     *
     * @return Partitions this into an array of vectors so that v[i].get(j) is
     * the ith element in the jth vector.
     */
    public Vector[] vecPartition() {
        Vector[] parts = new Vector[getSubVecDim()];
        Arrays.setAll(parts, i -> get(i));
        return parts;
    }

    /**
     * The element at the ith index of each subVector.
     *
     * @param i The index of the desired elements.
     * @return An array, a such that a_j is the ith element of the jth array.
     */
    public Vector get(int i) {
        return new Vector(handle, data.subArray(i * colDist), data.stride);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public VectorsStride fill(double val) {
        super.fill(val);
        return this;
    }

    /**
     * @see MatricesStride#add(boolean, double, JCudaWrapper.algebra.MatricesStride, double, JCudaWrapper.array.DArray)
     */
    public VectorsStride add(boolean transpose, double timesToAdd, VectorsStride toAdd, double timesThis, DArray workSpace) {
        super.add(transpose, timesToAdd, toAdd, timesThis, workSpace);
        return this;
    }

    /**
     * A contiguous subset of the subvectors in this set.
     *
     * @param start The index of the first subvector.
     * @param length The number of subvectors.
     * @return The subset.
     */
    @Override
    public VectorsStride subBatch(int start, int length) {
        return new VectorsStride(
                handle,
                data.subBatch(start, length),
                inc(),
                dim(),
                data.stride,
                length
        );
    }

    @Override
    public void close() {
        data.close();
    }

    /**
     * The data underlying these vectors.
     *
     * @return The data underlying these vectors.
     */
    public DStrideArray dArray() {
        return data;
    }

    /**
     * The increments between elements of the subvectors. This is the column
     * distance.
     *
     * @return The increments between elements of the subvectors. This is the
     * column distance.
     */
    public int inc() {
        return super.getColDist(); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/OverriddenMethodBody
    }

    /**
     * Changes each element x to x squared.
     *
     * @param normsGoHere This array will hold the norm of each vector.
     * @return
     */
    public Vector norms(DArray normsGoHere) {
        Vector norms = new Vector(handle, normsGoHere, 1);
        norms.addBatchVecVecMult(1, this, this, 0);
        KernelManager.get("sqrt").mapToSelf(handle, norms);
        return norms;
    }

    /**
     * Turns these vectors into unit vectors, and then multiplies them by the magnitude.
     *
     * @param magnitude The magnitude that each vector will be stretched of
     * squished to have.
     * @param workSpace Should be 2 * batchSize in length.
     * @return this.
     */
    public VectorsStride setVectorMagnitudes(double magnitude, DArray workSpace) {
        Vector norms = norms(workSpace.subArray(0, data.batchSize));

        Vector normsInverted = new Vector(handle, workSpace.subArray(data.batchSize), 1)
                .fill(magnitude).ebeDivide(norms);

        multiply(normsInverted);

        return this;
    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                Vector vec = new Vector(hand, 1,2,3,4,99,  5,6,7,8,99,  9,10,11,12,99)) {
            
            int width = 3, height = 4, colDist = 5;
            
            VectorsStride rows = vec.subVectors(1, width, colDist, height);
            
            System.out.println(rows);//2,6,10
            
            
        }
    }

    /**
     * The vectors in this brought to the cpu.
     *    
     * @return each vector as a [row][column].
     */
    public double[][] copyToCPURows() {
        double[][] copy = new double[getBatchSize()][];

        Arrays.setAll(copy, i -> getVector(i).toArray());

        return copy;
    }

    @Override
    public String toString() {
        
        return Arrays.deepToString(this.copyToCPURows()).replace("],", "],\n");
    }

    /**
     * The number of elements in each vector. This is the width.
     *
     * @return The number of elements in each vector. This is the width.
     */
    public int dim() {
        return width;
    }
}
