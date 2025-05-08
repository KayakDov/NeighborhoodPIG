package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class PSingletonToD2d extends PSingletonTo2d implements PointToD2d{

    /**
     * Constructs a singleton from an array and a desired index within that
     * array.
     *
     * @param from The array the singleton is a subset of.
     * @param index The index of the desired singleton.
     */
    public PSingletonToD2d(PointToD2d from, int index) {
        super(from, index);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD2d copy(Handle handle) {
        PSingletonToD2d copy = new PSingletonToD2d(
                new PArray2dToD2d(1, 1, targetDim.entriesPerLine, targetDim.numLines), 
                0
        );
        copy.set(handle, this);
        return copy;
    }

    /**
     * Sets the pointer in this singleton to point to val.
     * @param handle
     * @param val The target of the pointer in this singleton.
     * @return this.
     */
    public PSingletonToD2d set(Handle handle, DArray2d val) {
        super.set(handle, val);
        
        return this;
    }
    
    /**
     * The array pointed to by the pointer held in this singleton.
     * @param hand
     * @return The array pointed to by the pointer held in this pointer.
     */
    @Override
    public DArray2d getVal(Handle hand){
        Pointer arrayAdress = new Pointer();
        Pointer hostToArrayAdress = Pointer.to(arrayAdress);
        get(hand, hostToArrayAdress);
        return new DArray2d(arrayAdress, targetDim.entriesPerLine, targetDim.numLines, targetLD.getVal(hand));
    }

    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD2d as1d() {
        return this;
    }

    /**
     * TODO: implement this as a 1x1 array.
     *
     * @deprecated
     */
    @Override
    public Array2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * TODO: implement this as a 1x1x1 array.
     *
     * @deprecated
     */
    @Override
    public Array3d as3d(int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
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
    public DArray2d[] get(Handle hand) {
        return new DArray2d[]{getVal(hand)};
    }
}
