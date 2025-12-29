package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.stream.IntStream;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public interface PointToI2d extends PointTo2d {

    /**
     * {@inheritDoc }
     */
    @Override
    public default int targetBytesPerEntry() {
        return Sizeof.INT;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default IArray2d[] get(Handle hand) {
        Pointer[] cpuPointerArray = getPointers(hand);

        int[] ld = targetLD().get(hand);

        return IntStream.range(0, size()).mapToObj(i
                -> new IArray2d(
                        cpuPointerArray[i],
                        targetDim().entriesPerLine,
                        targetDim().numLines,
                        ld[i]
                )
        ).toArray(IArray2d[]::new);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public default PSingletonToI2d get(int index) {
        return new PSingletonToI2d(this, index);
    }
    
    
    /**
     * {@inheritDoc }
     * TODO: can this be done on the gpu?  IMplement in To2d instead of ToD2d?
     */
    @Override
    public default PointToI2d initTargets(Handle hand){
        for(int i = 0; i < size(); i++)
            get(i).set(hand, new IArray2d(targetDim().entriesPerLine, targetDim().numLines));
        return this;
    }
}
