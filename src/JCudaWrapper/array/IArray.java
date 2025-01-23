package JCudaWrapper.array;

import java.util.Arrays;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import JCudaWrapper.resourceManagement.Handle;

/**
 * An array of integers.
 *
 * @author E. Dov Neimand
 */
public class IArray extends Array {

    /**
     * Constructs an empty array.
     *
     * @param from An array containing this array.
     * @param offset The start position of this subarray.
     * @param length The length of the array.
     */
    public IArray(Array from, int offset, int length) {
        super(from, offset, length);
    }
    
    
    /**
     * Constructs an empty array.
     *
     * @param length The length of the array.
     */
    public IArray(int length) {
        super(length, PrimitiveType.INT);
    }

    /**
     * 
     * @param handle
     * @param values The values to be stored in the cpu array,
     */
    public IArray(Handle handle, int ... values) {
        super(values.length, PrimitiveType.INT);
        set(handle, Pointer.to(values), 0, 0, values.length);
    }

    /**
     * {@inheritDoc}
     *
     * @param handle The handle.
     * @return A copy of this array.
     */
    @Override
    public IArray copy(Handle handle) {
        IArray copy = new IArray(length);
        copy.set(handle, pointer, length);
        return copy;
    }

    /**
     * Exports the contents of this array to the cpu array.
     *
     * @param handle The handle.
     * @param toCPUArray The cpu array into which will be copied the elements of
     * this array.
     * @param toStart The index in the array to start copying to.
     * @param fromStart The index in this array to start copying from.
     * @param n The number of elements to copy.
     */
    public void get(Handle handle, int[] toCPUArray, int toStart, int fromStart, int n) {
        
        if(toStart + n > toCPUArray.length) throw new ArrayIndexOutOfBoundsException();
        
        super.get(handle, Pointer.to(toCPUArray), toStart, fromStart, n);
    }
    
    /**
     * Exports the contents of this array to the cpu array.
     *
     * @param handle The handle.
     * @param fromStart The index in this array to start copying from.
     * @param n The number of elements to copy.
     * @return A subsection of this array of integers.
     */
    public int[] get(Handle handle, int fromStart, int n) {
        int[] toCPUArray = new int[n];
        super.get(handle, Pointer.to(toCPUArray), 0, fromStart, n);
        return toCPUArray;
    }
    
    /**
     * Exports the contents of this array to the cpu.
     *
     * @param handle The handle.
     * @return The contents of this array stored in a cpu array.
     */
    public int[] get(Handle handle) {
        
        int[] toCPUArray = new int[length];
        super.get(handle, Pointer.to(toCPUArray), 0, 0, length);
        handle.synch();
        return toCPUArray;
    }


    @Override
    public String toString() {
        try (Handle hand = new Handle()) {
            int[] cpuArray = new int[length];
            get(hand, cpuArray, 0, 0, length);
            return Arrays.toString(cpuArray);
        }
    }

    
}
