package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.ISingleton;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.PSingleton;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class PSingletonTo2d extends PSingleton implements PointerTo2d {

    public final TargetDim2d target;
    public final ISingleton pitch;

    /**
     * Constructs the singleton pointer to a 2d array.
     *
     * @param from The array the singleton is taken from.
     * @param index The index of the desired entry.
     */
    public PSingletonTo2d(PointerTo2d from, int index) {
        super(from, index);
        pitch = from.targetPitches().get(from.memIndex(index));
        target =  from.target();
    }    

    /**
     * Constructs an empty instance.
     * @param targetMetaData The meta data for the array that this will point to.
     * @param pitch The pitch for the array that this will point to.
     */
    public PSingletonTo2d(TargetDim2d targetMetaData, ISingleton pitch) {
        super();
        this.target = targetMetaData;
        this.pitch = pitch;
    }

    /**
     * Sets the pointer in this singleton to point to the proffered 2d array.
     * @param handle
     * @param val
     * @return 
     */
    public PSingletonTo2d set(Handle handle, Array2d val) {
        Pointer cpuPointer = P.to(val);
        super.set(handle, cpuPointer);
        pitch.set(handle, val.ld());
        return this;
    }       
    
    /**
     * {@inheritDoc }
     */
    @Override
    public IArray targetPitches() {
        return pitch;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d target() {
        return target;
    }
    
}
