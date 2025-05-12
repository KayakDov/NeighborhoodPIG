package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.stream.IntStream;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public interface PointToF2d extends PointTo2d{
    
    /**
     * {@inheritDoc }
     */
    @Override
    public default int targetBytesPerEntry() {
        return Sizeof.FLOAT;
    }
    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public default FArray2d[] get(Handle hand) {
        Pointer[] cpuPointerArray = getPointers(hand);

        int[] ld = targetLD().get(hand);

        return IntStream.range(0, size()).mapToObj(i
                -> new FArray2d(
                        cpuPointerArray[i],
                        targetDim().entriesPerLine,
                        targetDim().numLines,
                        ld[i]
                )
        ).toArray(FArray2d[]::new);
    }
}
