package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * A 1d array of pointers.
 *
 * @author dov
 */
public abstract class PArray1d extends Array1d implements PArray {

    /**
     * Constructs an empty array.
     *
     * @param numElements The number of elements pointed to.
     */
    public PArray1d(int numElements) {
        super(numElements, Sizeof.POINTER);
    }

    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     */
    public PArray1d(PArray1d src, int start, int length) {
        super(src, start, length, src.ld());
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
    public PArray1d(PArray1d src, int start, int length, int ld) {
        super(src, start, length, ld);
    }

    
    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        JCuda.cudaDeviceSynchronize();
        try (Handle hand = new Handle()) {
            return Arrays.toString(get(hand));
        }
    }

}
