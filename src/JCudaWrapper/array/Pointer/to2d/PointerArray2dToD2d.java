package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Pointer.to1d.PointerArray1dToD1d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Pointer.PointerArray2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class PointerArray2dToD2d extends PointerArray2d implements PointToD2d {

    private final TargetDim2d targetDim;
    private final IArray2d targetPitch;

    /**
     * Constructs the empty array.
     *
     * @param entriesPerLine The number of pointers per line of pointers.
     * @param numLines The number of lines of pointers.
     * @param pointedToEntPerLine The number of entries per line in the arrays
     * that are pointed to..
     * @param pointedToNumLines The number of lines in the arrays that are
     * pointed to.
     */
    public PointerArray2dToD2d(int entriesPerLine, int numLines, int pointedToEntPerLine, int pointedToNumLines) {
        super(entriesPerLine, numLines);
        targetDim = new TargetDim2d(pointedToEntPerLine, pointedToNumLines);
        targetPitch = new IArray2d(entriesPerLine, numLines);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PSingletonToD2d get(int index) {
        return new PSingletonToD2d(this, index);
    }

    /**
     * Sets the pointers in this gpu array to point to be the gpu arrays in the
     * proffered cpu array.
     *
     * @param handle The cntext.
     * @param srcCPUArrayOfArrays A cpu array of gpu arrays, a pointer to each
     * of which will be stored in this gpu array.
     * @return this.
     */
    public PointerArray2dToD2d set(Handle handle, Array2d... srcCPUArrayOfArrays) {
        super.set(handle, srcCPUArrayOfArrays);

        targetPitch.set(
                handle,
                Arrays.stream(srcCPUArrayOfArrays)
                        .mapToInt(array -> array.pitch())
                        .toArray()
        );
        return this;
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
    public int targetBytesPerEntry() {
        return Sizeof.DOUBLE;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public IArray targetPitches() {
        return targetPitch;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d target() {
        return targetDim;
    }

}
