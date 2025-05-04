package JCudaWrapper.array;

import JCudaWrapper.array.Pointer.to1d.PSingletonTo1d;
import JCudaWrapper.array.Pointer.to1d.PointerTo1d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaPitchedPtr;

/**
 * A one-dimensional array stored in GPU memory. This class provides methods for
 * allocating, accessing, and transferring data between host and device memory.
 *
 * 1d arrays have one element on each line and an increment of ld().
 *
 * @author E. Dov Neimand
 */
public abstract class Array1d implements Array {

    private final Pointer pointer;
    public final int bytesPerEntry;
    public final int size;
    public final int ld;

    /**
     * Creates a new empty array.
     *
     * @param size The number of elements in the array.
     * @param bytesPerElement The number of bytes each element uses.
     */
    public Array1d(int size, int bytesPerElement) {
        pointer = new CUdeviceptr();
        this.bytesPerEntry = bytesPerElement;
        opCheck(JCuda.cudaMalloc(pointer, size * bytesPerElement));
        this.size = size;
        Array.recordMemAloc(pointer);
        ld = 1;
    }
    
    /**
     * Loads the array from memory.
     * @param hand The context.
     * @param pSingTo1d The memory to be in this array.
     */
    public Array1d(Handle hand, PSingletonTo1d pSingTo1d){
        pointer = pSingTo1d.get(hand);
        bytesPerEntry = pSingTo1d.targetBytesPerEntry();
        size = pSingTo1d.targetSize();
        ld = 1;
    }

    /**
     * Creates a sub array from the given array.
     *
     * @param src The array copied from.
     * @param start The start index in the array copied from where this array
     * begins.
     * @param size The number of elements in this array.
     * @param ld The increment between elements of the src. If the src is
     * already incremented, then this is the rate at which to increment over
     * those increments.
     */
    public Array1d(Array src, int start, int size, int ld) {
        bytesPerEntry = src.bytesPerEntry();
        pointer = src.pointer().withByteOffset(start * bytesPerEntry);
        this.size = size;
        this.ld = src.ld() * ld;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Pointer pointer() {
        return pointer;
    }

    /**
     * Constructs a pitched pointer for this 1d array, so that this array can be
     * copied to or from by a multi dimensional array.
     *
     * @param entriesPerLine This should be the entries per line in the other array
     * @param linesPerLayer The number of lines in each layer of the other array.
     * 
     * @return A pitched pointer to this array.
     */
    public cudaPitchedPtr pitchedPointer(int entriesPerLine, int linesPerLayer) {
        cudaPitchedPtr cpp = new cudaPitchedPtr();
        cpp.ptr = pointer;
        cpp.pitch = entriesPerLine * bytesPerEntry;
        cpp.xsize = cpp.pitch;
        cpp.ysize = linesPerLayer;
        return cpp;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int bytesPerEntry() {
        return bytesPerEntry;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return size;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void get(Handle handle, Array dst) {
        if (ld() == 1 && entriesPerLine() != ld())
            opCheck(JCuda.cudaMemcpyAsync(
                    dst.pointer(),
                    pointer(),
                    size() * bytesPerEntry(),
                    cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    handle.getStream()
            ));
        throw new UnsupportedOperationException("This opration is only supported for ld = 1.  You have ld = " + ld() + " dst is a " + dst.getClass());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void get(Handle handle, Pointer dstCPUArray) {
        if (!hasPadding())
            opCheck(JCuda.cudaMemcpyAsync(
                    dstCPUArray,
                    pointer(),
                    size() * bytesPerEntry(),
                    cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    handle.getStream()
            ));
        else
            throw new UnsupportedOperationException("This opration is only supported for ld = 1.  You have ld = " + ld());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array1d set(Handle handle, Pointer srcCPUArray) {
        if (ld() == 1)
            opCheck(JCuda.cudaMemcpyAsync(
                    pointer(),
                    srcCPUArray,
                    size() * bytesPerEntry(),
                    cudaMemcpyKind.cudaMemcpyHostToDevice,
                    handle.getStream()
            ));
        else
            throw new UnsupportedOperationException("This opration is only supported for ld = 1.  You have ld = " + ld());
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    abstract public Array copy(Handle handle);

    /**
     * This array represented as a 2d array.
     *
     * @param entriesPerLine The number of entries on each line of the 2d array.
     * @return This array represented as a 2d array.
     */
    abstract public Array2d as2d(int entriesPerLine);

    /**
     * This array represented as a 2d array.
     *
     * @param entriesPerLine The number of entries on each line of the 2d array.
     * @param ld The number of elements that could fit between the first
     * elements of each row, if the padding were used.
     * @return This array represented as a 2d array.
     */
    abstract public Array2d as2d(int entriesPerLine, int ld);

    /**
     * This array represented as a 2d array.
     *
     * @param entriesPerLine The number of entries on each line of the 2d array.
     * @param linesPerLayer The number of lines on each layer of the 3d array.
     * @return This array represented as a 2d array.
     */
    abstract public Array3d as3d(int entriesPerLine, int linesPerLayer);

    /**
     * This array represented as a 2d array.
     *
     * @param entriesPerLine The number of entries on each line of the 2d array.
     * @param ld The number of elements that could fit between the first
     * @param linesPerLayer The number of lines on each layer of the 3d array.
     * @return This array represented as a 2d array.
     */
    abstract public Array3d as3d(int entriesPerLine, int ld, int linesPerLayer);

    /**
     * {@inheritDoc}
     */
    @Override
    public int ld() {
        return ld;
    }

    /**
     *
     * @param start The starting index of the sub array.
     * @param size The number of elements in the sub array.
     * @return A sub array of this array.
     */
    public abstract Array1d sub(int start, int size);

    /**
     *
     * @param start The starting index of the sub array.
     * @param size The number of elements in the sub array.
     * @param ld The increment between elements of the sub array.
     * @return A sub array of this array.
     */
    public abstract Array1d sub(int start, int size, int ld);
}
