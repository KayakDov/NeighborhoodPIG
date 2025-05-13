package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 *
 * @author dov
 */
public class PSingletonToF2d extends PSingletonTo2d implements PointToF2d{
    
    /**
     * Constructs a singleton from an array and a desired index within that
     * array.
     *
     * @param from The array the singleton is a subset of.
     * @param index The index of the desired singleton.
     */
    public PSingletonToF2d(PointToF2d from, int index) {
        super(from, index);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToF2d copy(Handle handle) {
        PSingletonToF2d copy = new PSingletonToF2d(
                new PArray2dToF2d(1, 1, targetDim.entriesPerLine, targetDim.numLines), 
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
    public PSingletonToF2d set(Handle handle, FArray2d val) {
        super.set(handle, val);
        
        return this;
    }
    
    /**
     * The array pointed to by the pointer held in this singleton.
     * @param hand
     * @return The array pointed to by the pointer held in this pointer.
     */
    @Override
    public FArray2d getVal(Handle hand){
        Pointer arrayAdress = new Pointer();
        Pointer hostToArrayAdress = Pointer.to(arrayAdress);
        get(hand, hostToArrayAdress);
        return new FArray2d(arrayAdress, targetDim.entriesPerLine, targetDim.numLines, targetLD.getVal(hand));
    }

    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToF2d as1d() {
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray2d[] get(Handle hand) {
        return new FArray2d[]{getVal(hand)};
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
    public PSingletonToF2d get(int index) {
        confirm(index == 0);
        return this;
    }
}
