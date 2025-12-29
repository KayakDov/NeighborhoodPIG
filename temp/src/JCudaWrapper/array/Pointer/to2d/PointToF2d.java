package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Double.DArray1d;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Pointer.to1d.PointTo1d;
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
    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public default PSingletonToF2d get(int index) {
        return new PSingletonToF2d(this, index);
    }
    
    /**
     * {@inheritDoc }
     * TODO: can this be done on the gpu?  IMplement in To2d instead of ToD2d?
     */
    @Override
    public default PointToF2d initTargets(Handle hand){
        for(int i = 0; i < size(); i++)
            get(i).set(hand, new FArray2d(targetDim().entriesPerLine, targetDim().numLines));
        return this;
    }
}
