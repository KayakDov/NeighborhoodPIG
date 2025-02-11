package JCudaWrapper.array;

import jcuda.Sizeof;

/**
 * A 1d array of pointers.
 *
 * @author dov
 */
public abstract class PointerArray1d extends Array1d implements PointerArray {
    
    public final int subArraySize;

    /**
     * Constructs an empty array.
     * @param numElements The number of elements pointed to.
     * @param subArraySize The number of elements in the sub arrays.
     */
    public PointerArray1d(int numElements, int subArraySize) {
        super(numElements, Sizeof.POINTER);
        this.subArraySize = subArraySize;
    }
    
    
    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     */
    public PointerArray1d(PointerArray1d src, int start, int length){
        super(src, start, length, src.ld());        
        subArraySize = src.subArraySize;
    }
    
    
    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     * @param ld The increment.
     */
    public PointerArray1d(PointerArray1d src, int start, int length, int ld){
        super(src, start, length, src.ld()*ld);
        subArraySize = src.subArraySize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int subArraySize() {
        return subArraySize;
    }

}
