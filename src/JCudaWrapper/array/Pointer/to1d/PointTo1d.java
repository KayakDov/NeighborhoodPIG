package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Pointer.PArray;

/**
 *
 * @author E. DOv Neimand
 */
public interface PointTo1d extends PArray{
    
    /**
     * The length of the array pointed to.
     * @return The length of the array pointed to.
     */
    public int targetSize();
    
}
