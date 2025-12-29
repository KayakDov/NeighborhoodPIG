package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Pointer.PSingleton;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;
import JCudaWrapper.array.Pointer.to1d.PointTo1d;

/**
 * A singleton to a pointer.
 *
 * @author E. Dov Neimand
 */
public abstract class PSingletonTo1d extends PSingleton {

    private int targetSize;
    
    /**
     * Constructor 
     * @param from This singleton will point to the indexed element of this array.
     * @param index T
     */
    public PSingletonTo1d(PointTo1d from, int index) {
        super(from, index);
        targetSize = from.targetSize();
    }    
    
    /**
     * Constructs a new empty singleton.
     * @param targetBytesPerEntry The size of each datum pointed to.
     */
    public PSingletonTo1d(){
        super();
    }

    
    /**
     * {@inheritDoc}
     */
    @Override
    public PSingletonTo1d set(Handle handle, Pointer lmnt) {
        super.set(handle, Pointer.to(lmnt));
        return this;
    }
    
    /**
     * The pointer in this singleton.
     * @param handle
     * @return 
     */
    @Override
    public abstract Array1d getVal(Handle handle);

    /**
     * The size of the sub array pointed to.
     * @return The size of the sub array pointed to.
     */
    public int targetSize() {
        return targetSize;
    }
}
