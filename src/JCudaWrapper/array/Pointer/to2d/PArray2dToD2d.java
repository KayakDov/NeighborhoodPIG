package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Pointer.to1d.PArray1dToD1d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class PArray2dToD2d extends PArray2dTo2d implements PointToD2d {

    /**
     * Constructs the empty array.
     *
     * @param entriesPerLine The number of pointers per line of pointers.
     * @param numLines The number of lines of pointers.
     * @param targetEntPerLine The number of entries per line in the arrays that
     * are pointed to..
     * @param targetNumLines The number of lines in the arrays that are pointed
     * to.
     */
    public PArray2dToD2d(int entriesPerLine, int numLines, int targetEntPerLine, int targetNumLines) {
        super(entriesPerLine, numLines, targetEntPerLine, targetNumLines);
    }

    /**
     * Creates an empty array with the same dimensions as this array.
     *
     * @return An empty array with the same dimensions as this array.
     */
    public PArray2dToD2d copyDim(Handle hand) {
        return new PArray2dToD2d(entriesPerLine(), linesPerLayer(), targetDim().entriesPerLine, targetDim().numLines).initTargets(hand);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PArray2dToD2d initTargets(Handle hand) {
        PointToD2d.super.initTargets(hand); 
        return this;
    }
    
    

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD2d get(int indexInLine, int lineNumber) {
        return get(lineNumber * entriesPerLine() + indexInLine);
    }

    /**
     * @deprecated
     */
    @Override
    public PArray2dToD2d copy(Handle handle) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * @deprecated
     */
    @Override
    public PArray1dToD1d as1d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * @deprecated
     */
    @Override
    public PArray2dToD2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * This operation is not supported and will throw an unsupported operation
     * exception..
     *
     * @deprecated
     */
    @Override
    public Array3d as3d(int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Multiplies every element in the target arrays by the scalar.
     * @param handle
     * @param scalar
     */
    public void setProduct(Handle handle, double scalar) {
        Kernel.run("multiplyScalar", handle, targetSize(),
                new PArray2dTo2d[]{this},
                new Dimensions(handle, targetDim().entriesPerLine, targetDim().numLines, entriesPerLine(), linesPerLayer()),
                P.to(scalar)
        );
        
    }
}
