package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import java.util.function.IntUnaryOperator;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.apache.commons.math3.exception.DimensionMismatchException;

/**
 *
 * @author dov
 */
public interface PointerArray extends Array {

    public int subArraySize();
    
    /**
     * Sets an index of this array.
     *
     * @param handle The handle.
     * @param array The array whose pointer is to be placed at the given index.
     * @param index The index to place the array at.
     */
    public default void set(Handle handle, Array array, int index) {
        if (array.size() != subArraySize())
            throw new DimensionMismatchException(array.size(), subArraySize());

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
    public default PointerArray set(Handle handle, Array source, IntUnaryOperator generator) {
        CUdeviceptr[] pointers = new CUdeviceptr[size()];
        Arrays.setAll(pointers, i -> source.pointer(generator.applyAsInt(i)));
        set(handle, Pointer.to(pointers));
        return this;
    }    

}
