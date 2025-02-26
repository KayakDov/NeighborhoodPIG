package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * A singleton to a pointer.
 *
 * @author E. Dov Neimand
 */
public abstract class PSingleton extends Singleton {

    private int subArraySize;
    
    /**
     * Constructor 
     * @param from This singleton will point to the indexed element of this array.
     * @param index T
     */
    public PSingleton(PointerArray1d from, int index) {
        super(from, index);
        subArraySize = from.subArraySize;
    }
    
    /**
     * Constructs a new empty singleton.
     */
    public PSingleton(){
        super(Sizeof.POINTER);
    }

    
    /**
     * {@inheritDoc}
     */
    @Override
    public PSingleton set(Handle handle, Pointer lmnt) {
        super.set(handle, Pointer.to(lmnt));
        return this;
    }
    
    /**
     * The pointer in this singleton.
     * @param handle
     * @return 
     */
    public Pointer get(Handle handle){
        Pointer[] arrayOfPointer = new Pointer[1];
        get(handle, Pointer.to(arrayOfPointer));
        return arrayOfPointer[0];
    }

    /**
     * The size of the sub array pointed to.
     * @return The size of the sub array pointed to.
     */
    public int subArraySize() {
        return subArraySize;
    }
}
