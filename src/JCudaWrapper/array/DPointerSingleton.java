package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author dov
 */
public class DPointerSingleton extends PSingleton implements DPointerArray{

    /**
     * The first element of the array.
     * @param src The array the singleton is a sub array of.
     * @param index THe index of the desired element.
     */
    public DPointerSingleton(PointerArray1d src, int index) {
        super(src, index);
    }

    /**
     * An empty singleton.
     */
    public DPointerSingleton(){
        super();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DPointerSingleton set(Handle handle, Array from) {
        super.set(handle, from); 
        return this;
    }    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public DPointerSingleton copy(Handle handle) {
        return new DPointerSingleton().set(handle, this);
    }
    
}
