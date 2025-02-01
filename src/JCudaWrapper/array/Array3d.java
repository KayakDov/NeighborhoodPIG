package JCudaWrapper.array;

import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Cube;
import java.util.HashSet;
import jcuda.runtime.cudaExtent;
import jcuda.runtime.cudaMemcpy3DParms;
import jcuda.runtime.cudaPitchedPtr;

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
 * The data is stored in a 3d row major format. The distance between the first
 * elements of each row, pointer.pitch >= width, is computed by JCuda.
 *
 * http://www.jcuda.org/tutorial/TutorialIndex.html#CreatingKernels
 *
 * TODO:Implement methods with JBLAS as an aulternative for when there's no gpu.
 *
 * @author E. Dov Neimand
 */
public abstract class Array3d implements Array {

//    private final Cleaner.Cleanable cleanable; TODO: Self cleaning removed since sub arrays may be intended to linger after parent arrays are destroyed or vice versa. A more nuanced approach is required.
    /**
     * The pointer to the array in gpu memory.
     */
    protected final cudaPitchedPtr pointer;
    /**
     * The length of the array.
     */
    public final int width, height, depth;
    public final int bytesPerEntry;

    private static int arrayCount = 0;
    public final int ID;
    public static HashSet<Integer> alocatedArrays = new HashSet<>(100);

    /**
     * Constructs an Array with the given GPU pointer, length, and element type.
     *
     * @param length The length of the array.
     * @param bytesPerEntry The type of elements in the array.
     *
     * @throws IllegalArgumentException if the pointer is null or length is
     * negative.
     */
    protected Array3d(int length, int bytesPerEntry) {
        this(1, length, 1, bytesPerEntry);
    }

    /**
     * Constructs a 3d array.
     *
     * @param height The height of the memory to be allocated.
     * @param width The width of the memory to be allocated.
     * @param depth The depth of the memory to be allocated.
     * @param bytesPerEntry The memory size in bytes of elements stored in this
     * array.
     */
    protected Array3d(int height, int width, int depth, int bytesPerEntry) {

        this.bytesPerEntry = bytesPerEntry;
        alocatedArrays.add(ID = arrayCount++);

        this.height = height;
        this.width = width;
        this.depth = depth;

        cudaExtent extent = new cudaExtent(width * bytesPerEntry, height, depth);
        cudaPitchedPtr pitchedPtr = new cudaPitchedPtr();
        pointer = pitchedPtr;
        int error = JCuda.cudaMalloc3D(pitchedPtr, extent);

        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("Opening a new array of type " + bytesPerEntry + " with " + height * width * depth + " elements.  cuda error " + cudaError.stringFor(error));
    }

    /**
     * Constructs a sub array.
     *
     * @param from The array this one is to be a sub array of.
     * @param cube The indices of the sub array.
     * @param index The index of the desired row, column, or layer. start.
     */
    protected Array3d(Array3d from, Cube cube, int index) {

        this.pointer = new cudaPitchedPtr();
        this.pointer.ptr = from.pointer(cube.minX, cube.minY, cube.minZ);

        pointer.pitch = from.pointer.pitch;
        pointer.xsize = width = Math.min(cube.maxX - cube.minX, from.width - cube.minX);
        pointer.ysize = height = Math.min(cube.maxY - cube.minY, from.height - cube.minY);
        depth = Math.min(cube.maxZ - cube.minZ, from.depth - cube.minZ);
        this.bytesPerEntry = from.bytesPerEntry;
        ID = from.ID;
    }

    /**
     * Returns a pointer to the element at the specified 3D coordinates in the
     * pitched memory.
     *
     * @param x The x-coordinate.
     * @param y The y-coordinate.
     * @param z The z-coordinate.
     * @return A Pointer pointing to the specified element in the pitched
     * memory.
     *
     * @throws ArrayIndexOutOfBoundsException if the coordinates are out of
     * bounds.
     */
    private Pointer pointer(int x, int y, int z) {
        return pointer.ptr.withByteOffset(z * pointer.pitch * height + y * pointer.pitch + x * bytesPerEntry);
    }

    private cudaMemcpy3DParms copyFromParams;
    private cudaMemcpy3DParms copyToParams;

    /**
     * Sets memoryParms if it needs to be set, along with its fields srcPointer,
     * and extent.
     */
    private cudaMemcpy3DParms copyFromParams() {
        if (copyFromParams == null) {
            copyFromParams = new cudaMemcpy3DParms();
            copyFromParams.srcPtr = pointer;
            copyFromParams.extent = new cudaExtent(width * bytesPerEntry, height, depth);
            return copyFromParams;
        } else return copyFromParams;
    }


    /**
     * Sets memoryParms if it needs to be set, along with its fields dstPointer,
     * and extent.
     */    
    private cudaMemcpy3DParms copyToParams(){
        if (copyToParams == null) {
            copyToParams = new cudaMemcpy3DParms();
            copyToParams.dstPtr = pointer;
            copyToParams.extent = new cudaExtent(width * bytesPerEntry, height, depth);
            return copyToParams;
        } else return copyToParams;
    }

    /**
     * Copies data from this GPU array to another GPU array. It is assumed that
     * the array being copied to and from are of the same dimensions. It may be
     * necessary to construct a new sub array to ensure this condition is met.
     *
     * @param to The destination GPU array.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    @Override
    public void get(Handle handle, Array to) {
        copyFromParams = copyFromParams();

        copyFromParams.kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice;
        copyFromParams.dstPtr = to.pitchedPointer();

        int error = JCuda.cudaMemcpy3DAsync(copyFromParams, handle.getStream());
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }
    
    /**
     * Copies data from this GPU array to a CPU array.
     *
     * @param toCPUArray The destination CPU array.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    @Override
    public void get(Handle handle, Pointer toCPUArray) {

        copyFromParams = copyFromParams();

        copyFromParams.kind = cudaMemcpyKind.cudaMemcpyDeviceToHost;
        copyFromParams.dstPtr = arrayPitchedPointer(toCPUArray);

        int error = JCuda.cudaMemcpy3DAsync(copyFromParams, handle.getStream());
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * Creates a pitched pointer for copying between this and a cpu array.
     * @param cpuArray A pointer to the cpu array.
     * @return A pitched pointer for copying between this and a cpu array.
     */
    private cudaPitchedPtr arrayPitchedPointer(Pointer cpuArray){
        cudaPitchedPtr pptr = new cudaPitchedPtr();
        pptr.ptr = cpuArray;
        pptr.pitch = (long) width * bytesPerEntry;
        pptr.xsize = width*bytesPerEntry;
        pptr.ysize = height;
        return pptr;
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

        copyToParams = copyToParams();

        copyToParams.kind = cudaMemcpyKind.cudaMemcpyHostToDevice;
        copyFromParams.srcPtr = arrayPitchedPointer(fromCPUArray);

        int error = JCuda.cudaMemcpy3DAsync(copyFromParams, handle.getStream());
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));

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
        checkUpperBound(width, maybeInBounds);
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
}
