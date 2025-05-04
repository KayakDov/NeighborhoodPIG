package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Pointer.to1d.PointerArray1dToD1d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Pointer.PointerArray2d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class PointerArray2dToD2d extends PointerArray2d implements PointToD2d{

    private final TargetDim2d pointTo;
    private final IArray2d pointToPitch;

    /**
     * Constructs the empty array.
     * @param entriesPerLine The number of pointers per line of pointers.
     * @param numLines The number of lines of pointers.
     * @param pointedToEntPerLine The number of entries per line in the arrays that are pointed to..
     * @param pointedToNumLines The number of lines in the arrays that are pointed to.
     */
    public PointerArray2dToD2d(int entriesPerLine, int numLines, int pointedToEntPerLine, int pointedToNumLines) {
        super(entriesPerLine, numLines);
        pointTo = new TargetDim2d(pointedToEntPerLine, pointedToNumLines);
        pointToPitch = new IArray2d(entriesPerLine, numLines);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD2d get(int index) {
        return new PSingletonToD2d(this, index);
    }

    
    /**
     * @deprecated 
     */
    @Override
    public PointerArray2dToD2d copy(Handle handle) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    
    /**
     * @deprecated 
     */
    @Override
    public PointerArray1dToD1d as1d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    
    /**
     * @deprecated 
     */
    @Override
    public PointerArray2dToD2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    
    /**
     * This operation is not supported and will throw an unsupported operation exception..
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
    public IArray targetPitches() {
        return pointToPitch;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d target() {
        return pointTo;
    }

    
}
