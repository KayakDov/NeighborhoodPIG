package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class IArray1d extends Array1d implements IArray{
    
    /**
     * Constructs an empty IArray.
     * @param numElements The number of elements.
     */
    public IArray1d(int numElements) {
        super(numElements, Sizeof.INT);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array copy(Handle handle) {
        return new IArray1d(size()).set(handle, this);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Array2d as2d(int entriesPerLine) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Array3d as3d(int entriesPerLine, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Array3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Singleton get(int index) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */    
    @Override
    public Array1d sub(int start, int length) {
        throw new UnsupportedOperationException("Not yet implemented.");
    }
    
    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d sub(int start, int length, int ld) {
        throw new UnsupportedOperationException("Not yet implemented.");
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
