package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class IArray2d extends Array2d implements IArray{

    /**
     * Constructs a 2 dimensional Iarray.
     * @param entriesPerLine THe number of entries in each line.
     * @param numLines The number of lines in the array.
     */
    public IArray2d(int entriesPerLine, int numLines) {
        super(entriesPerLine, numLines, Sizeof.INT);
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Singleton get(int index) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array copy(Handle handle) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d as1d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}
