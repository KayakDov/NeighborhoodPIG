package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array3d;
import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to1d.PArray1dToD1d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import jcuda.Pointer;

/**
 *
 * @author E. Dov Neimand
 */
public class P2dToF2d extends PArray2dTo2d implements PointToF2d {

    /**
     * Constructs the empty array.
     *
     * @param entriesPerLine The number of pointers per line of pointers.
     * @param numLines The number of lines of pointers.
     * @param targetEntPerLine The number of entries per line in the arrays that
     * are pointed to..
     * @param targetNumLines The number of lines in the arrays that are pointed
     * to.
     * @param initializeTargets Leave null to not initialize targets. Otherwise,
     * this context is used to initialize targets.
     */
    public P2dToF2d(int entriesPerLine, int numLines, int targetEntPerLine, int targetNumLines, Handle initializeTargets) {
        super(entriesPerLine, numLines, targetEntPerLine, targetNumLines, initializeTargets);
    }

    /**
     * Creates an empty array with the same dimensions as this array.
     *
     * @param initializeTargets Leave null to not initialize targets. Otherwise,
     * this context is used to initialize targets.
     *
     * @return An empty array with the same dimensions as this array.
     */
    public P2dToF2d copyDim(Handle initializeTargets) {
        return new P2dToF2d(entriesPerLine(), linesPerLayer(), targetDim().entriesPerLine, targetDim().numLines, initializeTargets);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToF2d get(int indexInLine, int lineNumber) {
        if (indexInLine >= entriesPerLine() || lineNumber >= linesPerLayer())
            throw new IndexOutOfBoundsException();
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
     * {@inheritDoc }
     */
    @Override
    public P2dToF2d initTargets(Handle hand) {
        PointToF2d.super.initTargets(hand);
        return this;
    }

    /**
     * Multiplies every element in the target arrays by the scalar.
     *
     * @param handle
     * @param scalar
     * @param km
     */
    public void scale(Handle handle, float scalar) {

        try (Dimensions dims = new Dimensions(handle, targetDim().entriesPerLine, targetDim().numLines, entriesPerLine(), linesPerLayer())) {
            handle.runKernel("multiplyScalar", 
                    deepSize(),
                    new PArray2dTo2d[]{this},
                    dims,
                    P.to(scalar)
            );
        }
    }

    
    /**
     * All the elements pointed to  by a line of this array.
     * @param hand The context.
     * @param lineIndex The index of the desired line.
     * @return All the elements pointed to  by a line of this array.
     */
    public float[] getLine(Handle hand, int lineIndex){
        
        float[] line = new float[targetSize() * entriesPerLine()];
        
        for(int i = 0; i < entriesPerLine(); i++){
            Pointer pLine = Pointer.to(line).withByteOffset(bytesPerEntry * targetSize() * i);
            get(lineIndex, i).getVal(hand).get(hand, pLine);
        }
        
        return line;
    }
}
