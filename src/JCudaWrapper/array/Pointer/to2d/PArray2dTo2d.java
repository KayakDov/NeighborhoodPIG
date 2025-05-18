package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Pointer.PArray2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class PArray2dTo2d extends PArray2d implements PointTo2d {
    
    
    private final TargetDim2d targetDim;
    private final IArray2d targetLD;

    
    /**
     * Constructs the empty array.
     *
     * @param entriesPerLine The number of pointers per line of pointers.
     * @param numLines The number of lines of pointers.
     * @param targetEntPerLine The number of entries per line in the arrays
     * that are pointed to..
     * @param targetNumLines The number of lines in the arrays that are
     * pointed to.
     */
    public PArray2dTo2d(int entriesPerLine, int numLines, int targetEntPerLine, int targetNumLines) {
        super(entriesPerLine, numLines);
        targetDim = new TargetDim2d(targetEntPerLine, targetNumLines);
        targetLD = new IArray2d(entriesPerLine, numLines);
    }
    
    
    /**
     * Sets the pointers in this gpu array to point to be the gpu arrays in the
     * proffered cpu array.
     *
     * @param handle The cntext.
     * @param srcCPUArrayOfArrays A cpu array of gpu arrays, a pointer to each
     * of which will be stored in this gpu array.
     * @return this.
     */
    public PArray2dTo2d set(Handle handle, Array2d... srcCPUArrayOfArrays) {
        super.set(handle, srcCPUArrayOfArrays);

        targetLD().set(
                handle,
                Arrays.stream(srcCPUArrayOfArrays)
                        .mapToInt(array -> array.ld())
                        .toArray()
        );
        return this;
    }
    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public IArray2d targetLD() {
        return targetLD;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d targetDim() {
        return targetDim;
    }
}
