package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Pointer.PArray;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.runtime.JCuda;

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



    /**
     * {@inheritDoc }
     */
    @Override
    public default long totalMemoryUsed() {
        return targetSize() * targetBytesPerEntry() * size() + ld() * linesPerLayer() * bytesPerEntry();
    }    
}
