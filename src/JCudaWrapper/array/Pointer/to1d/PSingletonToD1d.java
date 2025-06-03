package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray1d;
import JCudaWrapper.array.Pointer.to1d.PSingletonTo1d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class PSingletonToD1d extends PSingletonTo1d implements PointToD1d {

    /**
     * The first element of the array.
     *
     * @param src The array the singleton is a sub array of.
     * @param index THe index of the desired element.
     */
    public PSingletonToD1d(PointToD1d src, int index) {
        super(src, index);
    }

    /**
     * An empty singleton.
     */
    public PSingletonToD1d() {
        super();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD1d set(Handle handle, Array from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD1d copy(Handle handle) {
        return new PSingletonToD1d().set(handle, this);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD1d as1d() {
        return this;
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

    /**
     * {@inheritDoc }
     */
    @Override
    public int targetSize() {
        return super.targetSize();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int targetBytesPerEntry() {
        return Sizeof.DOUBLE;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d[] get(Handle hand) {

        return new DArray1d[]{getVal(hand)};
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d getVal(Handle handle) {
        Pointer arrayAddress = new Pointer();
        Pointer toHostArray = Pointer.to(arrayAddress);
        get(handle, toHostArray);
        return new DArray1d(arrayAddress, targetSize());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d targetDim() {
        return new TargetDim2d(1, size());
    }


    
}
