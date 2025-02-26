package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.PointerArray;
import JCudaWrapper.array.Pointer.PointerArray1d;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.array.StrideArray;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E.Dov Neimand
 */
public class DPointerArray1d extends PointerArray1d implements PointerArray{
    
    /**
     * Constructs an array of double pointers.
     * @param subArraySize The length of the arrays pointed to.
     * @param size The number of pointers in this array.
     */
    public DPointerArray1d(int subArraySize, int size) {
        super(size, subArraySize);
    }

    /**
     * Constructs this array from a stride array. A pointer in this array will point
     * to the beginning of each sub array of the stride array.
     * @param handle The context
     * @param dsa The array to be pointed to.
     */
    public <DStrideArray extends StrideArray, DArray> DPointerArray1d(Handle handle, DStrideArray dsa){
        super(dsa.batchSize(), dsa.subArraySize());
        Kernel.run("genPtrs", handle, dsa.batchSize(), dsa, P.to(dsa.strideLines()), P.to(this), P.to(1));
    }
    
    
    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     */
    public DPointerArray1d(DPointerArray1d src, int start, int length){
        super(src, start, length);
    }
    
    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param length The number of elements in this array.
     */
    public DPointerArray1d(DPointerArray1d src, int start, int length, int ld){
        super(src, start, length, ld);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Singleton get(int index) {
        return new DPointerSingleton(this, index);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DPointerArray1d set(Handle handle, Array from) {
        super.set(handle, from); 
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public DPointerArray1d copy(Handle handle) {
        return new DPointerArray1d(subArraySize, size())
                .set(handle, this);
        
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d(int entriesPerLine) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int entriesPerLine, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Array1d sub(int start, int size) {
        return new DPointerArray1d(this, start, size);
    }

    @Override
    public Array1d sub(int start, int size, int ld) {
        return new DPointerArray1d(this, start, size, ld);
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d as1d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception.  TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    
}
