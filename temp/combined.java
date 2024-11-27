
package main;

import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

public class TestStuff {
    
    public static void main(String[] args) {
        
        int height =70, width = 90;
        
        System.out.println("Memory allocated = " + width * height);
                
        try(Handle hand = new Handle(); DArray array = DArray.empty(height*width)){            
        
            DStrideArray dsa = array.getAsBatch(1, height*(width - 1) + 1, height);
            
            for(int i = 0; i < height; i++){
                System.out.println("\n\ni = " + i); 
                
                DArray da = dsa.getBatchArray(i);
                
                System.out.println(" has length " + da.length + " and is " + da.toString().substring(0, 100));
            }
        }
    }
    
}



package JCudaWrapper.array;

import static JCudaWrapper.array.Array.checkNull;
import static JCudaWrapper.array.DArray.cpuPointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverEigMode;
import jcuda.jcusolver.gesvdjInfo;
import JCudaWrapper.resourceManagement.Handle;
import static JCudaWrapper.array.Array.checkPositive;
import jcuda.runtime.cudaError;

/**
 * A class for a batch of consecutive arrays.
 *
 * @author E. Dov Neimand
 */
public class DStrideArray extends DArray {

public final int stride, batchSize, subArrayLength;

    /**
     * The constructor. Make sure batchSize * strideSize is less than length 
     * @param p A pointer to the first element. 
     * from the first element of one subsequence to the first element of the next.      
     * @param strideSize The number of elements between the first element of each subarray. 
     * @param batchSize The number of strides. @param subArrayLength The length of e
     * @param subArrayLength The length of each sub arrau/
     */
    protected DStrideArray(CUdeviceptr p, int strideSize, int subArrayLength, int batchSize) {
        super(p, totalDataLength(strideSize, subArrayLength, batchSize));
        this.stride = strideSize;
        this.subArrayLength = subArrayLength;
        this.batchSize = batchSize;
    }

    /**
     * The constructor. Make sure batchSize * strideSize is less than length 
     * @param array The underlying data. 
     * @param strideSize The distance from the
     * first element of one subsequence to the first element of the next. 
     * @param batchSize The number of strides. 
     * @param subArrayLength The length of each sub array.
     */
    public DStrideArray(DArray array, int strideSize, int batchSize, int subArrayLength) {
        this(array.pointer, strideSize, subArrayLength, batchSize);
    }

    /**
     * The number of sub arrays.
     *
     * @return The number of sub arrays.
     */
    /**
     * The number of sub arrays.
     *
     * @return The number of sub arrays.
     */
    public int batchCount() {
        return batchSize;
    }


    /**
     * The length of the array.
     *
     * @param batchSize The number of elements in the batch.
     * @param strideSize The distance between the first elements of each batch.
     * @param subArrayLength The length of each subArray.
     * @return The minimum length to hold a batch described by these paramters.
     */
    public static int totalDataLength(int strideSize, int subArrayLength, int batchSize) {
        return strideSize * (batchSize - 1) + subArrayLength;
    }

    /**
     * An empty batch array.
     *
     * @param batchSize The number of subsequences.
     * @param strideSize The size of each subsequence.
     * @param subArrayLength The length of each sub arrays.
     * @return An empty batch array.
     */
    public static DStrideArray empty(int batchSize, int strideSize, int subArrayLength) {

        return new DStrideArray(
                Array.empty(totalDataLength(strideSize, subArrayLength, batchSize), PrimitiveType.DOUBLE),
                strideSize,
                subArrayLength, batchSize
        );
    }
    

    /**
     * A sub batch of this batch.
     *
     * @param start The index of the first sub array.
     * @param length The number of sub arrays in this array. Between 0 and batch
     * size.
     * @return A subbatch.
     */
    public DStrideArray subBatch(int start, int length) {
        return subArray(start * stride, totalDataLength(stride, subArrayLength, length)).getAsBatch(stride, subArrayLength, length);
    }
    
    /**
     * Gets the sub array at the given batch index (not to be confused with indices in the underlying array.)
     * @param i The batch index of the desired array: batch index = stride * i     
     * @return The member of the batch at the given batch index.
     */
    public DArray getBatchArray(int i){
        if(i >= batchSize) throw new ArrayIndexOutOfBoundsException();
    
        System.out.println("JCudaWrapper.array.DStrideArray.getBatchArray() The start index is " + stride * i + " the end index is " + (stride*i + subArrayLength));
        
        return subArray(stride*i, subArrayLength);
    }

    @Override
    public DStrideArray copy(Handle handle) {
        return super.copy(handle).getAsBatch(stride, subArrayLength, batchSize);
    }    
}



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
     * @param n The number of elements to export.
     * @return A CPU array containing the exported portion.
     * @throws IllegalArgumentException if fromStart or length is out of bounds.
     */
    public double[] get(Handle handle, int fromStart, int n) {
        double[] export = new double[n];
        handle.synch();
        get(handle, export, 0, fromStart, n);
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
        else 
            for (int i = 0; i < length; i++)
                get(handle, to, i * toInc + toStart, i * fromInc + fromStart, 1);
        
        handle.synch();
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

    @Override
    public String toString() {
        JCuda.cudaDeviceSynchronize();
        try (Handle handle = new Handle()) {
            return Arrays.toString(get(handle));
        }
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
}
