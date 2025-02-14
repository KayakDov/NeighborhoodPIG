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

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d as1d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}
