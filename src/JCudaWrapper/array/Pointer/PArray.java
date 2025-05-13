package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
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
     * {@inheritDoc }
     */
    @Override
    public default void close() {
        
        JCuda.cudaDeviceSynchronize();//TODO:This is a bit lazy.  Better to get a desired handle into here.

        try (Handle hand = new Handle()) {
            Kernel.run("deepFree", hand, ld() * size() / entriesPerLine(), this);//note, calling this method will give a false indication of memory leaks, since pointers allocated here and not being removed from the list of alocated memory.
        }
        Array.super.close();
    }

}
