package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Pointer.PointerArray;

/**
 *
 * @author E. DOv Neimand
 */
public interface PointerTo1d extends PointerArray{
    
    /**
     * The length of the array pointed to.
     * @return The length of the array pointed to.
     */
    public int targetSize();
}
