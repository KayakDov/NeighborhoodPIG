package JCudaWrapper.array.Int;

import JCudaWrapper.array.Array;
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
    public default IArray set(Handle handle, int... srcCPU) {
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

    /**
     * {@inheritDoc}
     */
    @Override
    public default IArray1d as1d() {
        return new IArray1d(this, 0, size(), ld());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default IArray2d as2d() {
        return new IArray2d(this, entriesPerLine());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public default IArray3d as3d(int linesPerLayer) {
        return new IArray3d(this, entriesPerLine(), linesPerLayer);
    }

    /**
     * Gets the singleton at the desired index.
     * @param index The index of the desired singleton.
     * @return The singleton at the desired index.
     */
    public default ISingleton get(int index){
         return new ISingleton(this, index);
    }
}
