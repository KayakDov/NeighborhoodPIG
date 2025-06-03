package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 *
 * @author dov
 */
public class PSingletonToI2d extends PSingletonTo2d implements PointToI2d{

    /**
     * Constructs a singleton from an array and a desired index within that
     * array.
     *
     * @param from The array the singleton is a subset of.
     * @param index The index of the desired singleton.
     */
    public PSingletonToI2d(PointToI2d from, int index) {
        super(from, index);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToI2d copy(Handle handle) {
        PSingletonToI2d copy = new PSingletonToI2d(
                new PArray2dToI2d(1, 1, targetDim.entriesPerLine, targetDim.numLines, handle), 
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
    public PSingletonToI2d set(Handle handle, IArray2d val) {
        super.set(handle, val);
        
        return this;
    }
    
    /**
     * The array pointed to by the pointer held in this singleton.
     * @param hand
     * @return The array pointed to by the pointer held in this pointer.
     */
    @Override
    public IArray2d getVal(Handle hand){
        Pointer arrayAdress = new Pointer();
        Pointer hostToArrayAdress = Pointer.to(arrayAdress);
        get(hand, hostToArrayAdress);
        return new IArray2d(arrayAdress, targetDim.entriesPerLine, targetDim.numLines, targetLD.getVal(hand));
    }

    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToI2d as1d() {
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public IArray2d[] get(Handle hand) {
        return new IArray2d[]{getVal(hand)};
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
    public PSingletonToI2d get(int index) {
        confirm(index == 0);
        return this;
    }    
}
