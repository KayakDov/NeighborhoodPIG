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
        
    protected TargetDim2d targetDim;
    protected IArray2d targetLD;
    
    /**
     * Constructs the empty array.
     *
     * @param entriesPerLine The number of pointers per line of pointers.
     * @param numLines The number of lines of pointers.
     * @param targetEntPerLine The number of entries per line in the arrays
     * that are pointed to.
     * @param targetNumLines The number of lines in the arrays that are
     * pointed to.
     * @param initializeTargets Leave null to not initialize targets.  Otherwise the handle is used for that.
     */
    public PArray2dTo2d(int entriesPerLine, int numLines, int targetEntPerLine, int targetNumLines, Handle initializeTargets) {
        super(entriesPerLine, numLines);
        targetLD = new IArray2d(entriesPerLine, numLines);
        targetDim = new TargetDim2d(targetEntPerLine, targetNumLines);
        if(initializeTargets != null) initTargets(initializeTargets);
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

        int[] cpuTargetLD = Arrays.stream(srcCPUArrayOfArrays)
                        .mapToInt(array -> array.ld())
                        .toArray();
        
        targetLD().set(handle, cpuTargetLD);
        return this;
    }
    
    


    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d targetDim() {
        return targetDim;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public IArray targetLD() {
        return targetLD;
    }
    
    /**
     *{@inheritDoc }
     */
    @Override
    public void close(){
        targetLD.close();
        super.close();
    }
}

