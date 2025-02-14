package JCudaWrapper.array;

import java.util.Arrays;
import jcuda.Pointer;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 * An array of integers.
 *
 * @author E. Dov Neimand
 */
public interface IArray extends Array {

    /**
     * Exports the contents of this array to the cpu array.
     *
     * @param handle The handle.
     * @param dstCPU The cpu array into which will be copied the elements of
     * this array.
     */
    public default void get(Handle handle, int[] dstCPU) {

        get(handle, Pointer.to(dstCPU));
    }

    /**
     * Exports the contents of this array to the cpu array.
     *
     * @param handle The handle.
     * @return A subsection of this array of integers.
     */
    public default int[] get(Handle handle) {
        int[] toCPUArray = new int[size()];
        get(handle, toCPUArray);
        return toCPUArray;
    }

    /**
     * Sets the contents of this gpu array from the proffered cpu array.
     *
     * @param handle
     * @param srcCPU The cpu array whos contents will be copied to here.
     * @return this
     */
    public default IArray set(Handle handle, int[] srcCPU) {
        set(handle, Pointer.to(srcCPU));
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default int bytesPerEntry() {
        return Sizeof.INT;
    }
}
