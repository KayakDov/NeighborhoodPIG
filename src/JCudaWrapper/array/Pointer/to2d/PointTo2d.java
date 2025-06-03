package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Pointer.PArray;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import java.util.stream.IntStream;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 *
 * @author E. Dov Neimand
 */
public interface PointTo2d extends PArray {

    /**
     * An array that contains the pitch value of each pointed to array.
     *
     * @return An array that contains the pitch value of each pointed to array.
     */
    public IArray targetLD();    

    /**
     * {@inheritDoc}
     */
    @Override
    public default int targetSize() {
        return targetDim().size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public default void close() {
        targetLD().close();
        PArray.super.close();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default long totalMemoryUsed() {
        long memoryUsed = targetLD().totalMemoryUsed() + 
                + ld() * linesPerLayer() * bytesPerEntry();

        JCuda.cudaDeviceSynchronize();

        try (Handle hand = new Handle()) {
            
            int[] lds = targetLD().get(hand);
            
            for (int ld :lds) memoryUsed += ld * targetDim().numLines * targetBytesPerEntry();
        }

        return memoryUsed;
    }

}
