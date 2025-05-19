package JCudaWrapper.array.Int;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray;
import JCudaWrapper.array.Double.DArray1d;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;

/**
 *
 * @author E. Dov Neimand
 */
public class IArray1d extends Array1d implements IArray {

    /**
     * Constructs an empty IArray.
     *
     * @param numElements The number of elements.
     */
    public IArray1d(int numElements) {
        super(numElements, Sizeof.INT);
    }
    
    /**
     * Constructs a 1d sub array of the proffered array. If the array copied
     * from is not 1d, then depending on the length, this array may include
     * pitch.
     *
     * @param src The array to be copied from.
     * @param start The start index of the array.
     * @param size The length of the array.
     */
    public IArray1d(IArray src, int start, int size) {
        super(src, start, size, 1);
    }
    
    /**
     * Sets the gpu array from a cpu array.
     * @param handle The context.
     * @param elements The elements to be copied to the gpu.
     */
    public IArray1d(Handle handle, int... elements) {
        this(elements.length);
        set(handle, elements);
    }

    /**
     * Constructs a 1d sub array of the proffered array.If the array copied from
     * is not 1d, then depending on the length, this array may include pitch.
     *
     * @param src The array to be copied from.
     * @param start The start index of the array.
     * @param size The length of the array.
     * @param ld The increment between elements.
     */
    public IArray1d(IArray src, int start, int size, int ld) {
        super(src, start, size, ld);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public IArray1d set(Handle handle, Array from) {
        super.set(handle, from);
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public IArray1d copy(Handle handle) {
        return new IArray1d(size()).set(handle, this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int[] get(Handle handle) {

        int[] cpuArray = new int[size()];

        if (!hasPadding()) get(handle, Pointer.to(cpuArray));
        else try (IArray1d temp = new IArray1d(size())) {
            get(handle, temp);
            temp.get(handle, Pointer.to(cpuArray));
        }

        return cpuArray;

    }
    
    /**
     * Copies from here to there with increments.
     *
     * @param handle The context.
     * @param dst Copy to here.
     */
    public void get(Handle handle, IArray1d dst) {
        opCheck(JCublas2.cublasScopy(
                handle.get(),
                Math.min(size(), dst.size()),
                pointer(),
                ld(),
                dst.pointer(),
                dst.ld()
        ));
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public IArray1d set(Handle handle, int... srcCPU) {
        IArray.super.set(handle, srcCPU);
        return this;
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d(int entriesPerLine) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int entriesPerLine, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        JCuda.cudaDeviceSynchronize();
        try (Handle handle = new Handle()) {
            return Arrays.toString(get(handle));
        }
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d sub(int start, int length) {
        throw new UnsupportedOperationException("Not yet implemented.");
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d sub(int start, int length, int ld) {
        throw new UnsupportedOperationException("Not yet implemented.");
    }
}
