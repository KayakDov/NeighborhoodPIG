package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Pointer.PArray;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import java.util.stream.IntStream;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public interface PointToD2d extends PointTo2d{
    
    /**
     * {@inheritDoc }
     */
    @Override
    public default int targetBytesPerEntry() {
        return Sizeof.DOUBLE;
    }
        
    /**
     * {@inheritDoc }
     */
    @Override
    public default DArray2d[] get(Handle hand) {
        Pointer[] cpuPointerArray = getPointers(hand);

        int[] ld = targetLD().get(hand);

        return IntStream.range(0, size()).mapToObj(i
                -> new DArray2d(
                        cpuPointerArray[i],
                        targetDim().entriesPerLine,
                        targetDim().numLines,
                        ld[i]
                )
        ).toArray(DArray2d[]::new);
    }
    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public default PSingletonToD2d get(int index) {
        return new PSingletonToD2d(this, index);
    }
    
    /**
     * {@inheritDoc }
     * TODO: can this be done on the gpu?  IMplement in To2d instead of ToD2d?
     */
    @Override
    public default PointTo2d initTargets(Handle hand){
        for(int i = 0; i < size(); i++)
            get(i).set(hand, new DArray2d(targetDim().entriesPerLine, targetDim().numLines));
        return this;
    }
}
