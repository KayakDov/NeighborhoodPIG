package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Pointer.to2d.PointTo2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import java.util.function.IntUnaryOperator;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;

/**
 *
 * @author dov
 */
public interface PArray extends Array {

    /**
     * Information about the arrays pointed to.
     *
     * @return Information about the arrays pointed to.
     */
    public int targetBytesPerEntry();

    /**
     * Sets an index of this array.
     *
     * @param handle The handle.
     * @param array The array whose pointer is to be placed at the given index.
     * @param index The index to place the array at.
     */
    public default void set(Handle handle, Array array, int index) {

        get(index).set(handle, array.pointer());
    }

    /**
     * Creates a host array of pointer objects that have not had memory
     * allocated. These objects are ready to have actual memory addressess
     * written to them from the device.
     *
     * @param length The length of the array.
     * @return An array of new pointer objects.
     */
    static CUdeviceptr[] emptyHostArray(int length) {
        CUdeviceptr[] array = new CUdeviceptr[length];
        Arrays.setAll(array, i -> new CUdeviceptr());

        return array;
    }

    /**
     * Sets the elements of this array to be pointers to sub sections of the
     * proffered array.
     *
     * @param handle The Handle.
     * @param source An array with sub arrays that are held in this array.
     * @param generator The index of the beginning of the sub array to be held
     * at the argument's index.
     * @return This.
     */
    public default PArray set(Handle handle, Array source, IntUnaryOperator generator) {
        CUdeviceptr[] pointers = new CUdeviceptr[size()];
        Arrays.setAll(pointers, i -> source.pointer(generator.applyAsInt(i)));
        set(handle, Pointer.to(pointers));
        return this;
    }

    /**
     * Gets the arrays pointed to by these arrays.
     *
     * @param hand The context.
     * @return Gets the arrays pointed to by these arrays.
     */
    public Array[] get(Handle hand);

    /**
     * Copies the gpu array of pointers to a cpu array of pointers.
     *
     * @param hand The context.
     * @return A cpu array of pointers that is a copy of this gpu array of
     * pointers.
     */
    default Pointer[] getPointers(Handle hand) {
        Pointer[] cpuPointerArray = new Pointer[size()];
        Arrays.setAll(cpuPointerArray, i -> new Pointer());

        Pointer hostToArray = Pointer.to(cpuPointerArray);
        get(hand, hostToArray);
        return cpuPointerArray;
    }

    /**
     * The size of the arrays being pointed to.
     *
     * @return The size of the arrays being pointed to.
     */
    public int targetSize();

    /**
     * The summation of the sizes of all the elements pointed to, assuming none
     * of the pointers are null.
     *
     * @return The summation of the sizes of all the elements pointed to,
     * assuming none of the pointers are null.
     */
    public default int deepSize() {
        return size() * targetSize();
    }

    /**
     * Frees all the pointers stored in this array, but does not free this
     * array.
     *
     * @param hand The context.
     */
    public default void clearPointers(Handle hand) {
        Pointer[] ptrs = getPointers(hand);
        for (Pointer ptr : ptrs) {
            if (!allocatedArrays.remove(ptr))
                System.err.println("Trying to remove a pointer " + ptr.toString() + " that does not exist or has already been removed.");
            JCuda.cudaFree(ptr);
        }
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default void close() {

        JCuda.cudaDeviceSynchronize();//TODO:This is a bit lazy.  Better to get a desired handle into here.

        try (Handle hand = new Handle()) {
            clearPointers(hand);
        }
        Array.super.close();
    }

    /**
     * Allocated empty arrays to all the null pointers.
     *
     * @param hand The context.
     * @return this.
     */
    public PArray initTargets(Handle hand);

    /**
     * The dimensions of the arrays, as a 2d array. If the array is 1d, then
     * best that the long dimension should be height, and the width should be 1.
     *
     * @return The number of entries per line in the arrays pointed to.
     */
    public TargetDim2d targetDim();
    
    /**
     * Describes any line array, except that there's nto pointer information.
     */
    public class TargetDim2d {

        public final int entriesPerLine;
        public final int numLines;

        /**
         * Constructor
         *
         * @param entriesPerLine The number of entries on each line.
         * @param numLines The number of lines.
         */
        public TargetDim2d(int entriesPerLine, int numLines) {
            this.entriesPerLine = entriesPerLine;
            this.numLines = numLines;
        }

        /**
         * The number of entries items.
         *
         * @return The number of entries.
         */
        public int size() {
            return entriesPerLine * numLines;
        }
    }
}
