package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray1d;
import JCudaWrapper.array.KernelManager;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.PArray;
import JCudaWrapper.array.Pointer.PArray1d;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.array.StrideArray;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author E.Dov Neimand
 */
public class PArray1dToD1d extends PArray1d implements PointToD1d{
        
    private final int targetSize;
    /**
     * Constructs an array of double pointers.
     * @param targetSize The length of the arrays pointed to.
     * @param size The number of pointers in this array.
     */
    public PArray1dToD1d(int targetSize, int size) {
        super(size);
        this.targetSize = targetSize;
    }

//    /**
//     * Constructs this array from a stride array.A pointer in this array will point
// to the beginning of each sub array of the stride array.
//     * @param <DStrideArray>
//     * @param <DArray>
//     * @param handle The context
//     * @param dsa The array to be pointed to.
//     */
//    public <DStrideArray extends StrideArray, DArray> PArray1dToD1d(Handle handle, DStrideArray dsa){
//        this(dsa.subArraySize(), dsa.batchSize());
//        Kernel.run(
//                "genPtrs", handle, 
//                dsa.batchSize(), dsa, 
//                P.to(dsa.strideLines()), 
//                P.to(this), 
//                P.to(1)
//        );
//    }
    
    
    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     */
    public PArray1dToD1d(PArray1dToD1d src, int start, int length){
        super(src, start, length);
        targetSize = src.targetSize;
    }
    
    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     */
    public PArray1dToD1d(PArray1dToD1d src, int start, int length, int ld){
        super(src, start, length, ld);
        targetSize = src.targetSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Singleton get(int index) {
        return new PSingletonToD1d(this, index);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public PArray1dToD1d set(Handle handle, Array from) {
        super.set(handle, from); 
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public PArray1dToD1d copy(Handle handle) {
        return new PArray1dToD1d(targetSize, size())
                .set(handle, this);
        
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Array1d sub(int start, int size) {
        return new PArray1dToD1d(this, start, size);
    }

    @Override
    public Array1d sub(int start, int size, int ld) {
        return new PArray1dToD1d(this, start, size, ld);
    }

    /**
     * @deprecated 
     */
    @Override
    public Array2d as2d(int entriesPerLine) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * @deprecated 
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * @deprecated 
     */
    @Override
    public Array3d as3d(int entriesPerLine, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * @deprecated 
     */
    @Override
    public Array3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public PArray1dToD1d as1d() {
        return new PArray1dToD1d(this, 0, size());
    }

    /**
     * @deprecated 
     */
    @Override
    public Array2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
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
    public int targetSize() {
        return targetSize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d[] get(Handle hand) {
            
        Pointer[] cpuArray = new Pointer[size()];

        if (!hasPadding()) get(hand, Pointer.to(cpuArray));
        
        else try (PArray1dToD1d temp = new PArray1dToD1d(targetSize(), size())) {
            get(hand, temp);
            temp.get(hand, Pointer.to(cpuArray));
        }

        return Arrays.stream(cpuArray).map(pointer -> new DArray1d(pointer, targetSize)).toArray(DArray1d[]::new);

    }

    /**
     * {@inheritDoc }
     */
    @Override
    public TargetDim2d targetDim() {
        return new TargetDim2d(1, size());
    }


}
