package JCudaWrapper.array;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class Singleton extends Array1d {

    /**
     * The first element in this array is this singleton.
     *
     * @param from The array this one is ato be a sub array of.
     * @param index The index of the desired singleton.
     */
    public Singleton(Array from, int index) {
        super(from, from.memIndex(index), 1, 1);
    }

    /**
     * An empty singleton.
     *
     * @param bytesPerElement The number of bytes.
     */
    public Singleton(int bytesPerElement) {
        super(1, bytesPerElement);
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d(int entriesPerLine) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int entriesPerLine, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    
    /**
     * {@inheritDoc }
     */
    @Override
    public Singleton get(int index) {
        confirm(index == 0);
        return this;
    }    

    /**
     * {@inheritDoc }
     */
    @Override
    public Array1d sub(int start, int length) {
        confirm(start == 0, length == 1);
        return this;
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d sub(int start, int size, int ld) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
    
}
