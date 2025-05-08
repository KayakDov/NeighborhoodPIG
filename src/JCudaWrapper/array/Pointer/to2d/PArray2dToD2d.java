package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Pointer.to1d.PArray1dToD1d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Pointer.PArray2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import java.util.stream.IntStream;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class PArray2dToD2d extends PArray2d implements PointToD2d {

    private final TargetDim2d targetDim;
    private final IArray2d targetLD;

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
    public PArray2dToD2d(int entriesPerLine, int numLines, int pointedToEntPerLine, int pointedToNumLines) {
        super(entriesPerLine, numLines);
        targetDim = new TargetDim2d(pointedToEntPerLine, pointedToNumLines);
        targetLD = new IArray2d(entriesPerLine, numLines);
    }

    /**
     * Creates an empty array with the same dimensions as this array.
     * @return An empty array with the same dimensions as this array.
     */
    public PArray2dToD2d copyDim(){
        return new PArray2dToD2d(entriesPerLine(), linesPerLayer(), targetDim.entriesPerLine, targetDim.numLines);
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
    public PArray2dToD2d set(Handle handle, Array2d... srcCPUArrayOfArrays) {
        super.set(handle, srcCPUArrayOfArrays);

        targetLD.set(
                handle,
                Arrays.stream(srcCPUArrayOfArrays)
                        .mapToInt(array -> array.ld())
                        .toArray()
        );
        return this;
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
    public int targetBytesPerEntry() {
        return Sizeof.DOUBLE;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public IArray targetLD() {
        return targetLD;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d targetDim() {
        return targetDim;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray2d[] get(Handle hand) {
        Pointer[] cpuArray = new Pointer[size()];
        Arrays.setAll(cpuArray, i -> new Pointer());
        
        Pointer hostToArray = Pointer.to(cpuArray);
        get(hand, hostToArray);

        int[] ld = targetLD.get(hand);

        return IntStream.range(0, size()).mapToObj(i
                -> new DArray2d(
                        cpuArray[i],
                        targetDim.entriesPerLine,
                        targetDim.numLines,
                        ld[i]
                )
        ).toArray(DArray2d[]::new);
    }

}
