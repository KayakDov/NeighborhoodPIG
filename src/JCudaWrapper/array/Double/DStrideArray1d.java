package JCudaWrapper.array.Double;

import JCudaWrapper.array.StrideArray;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class DStrideArray1d extends DArray1d implements StrideArray {

    public final int stride, batchSize, subArraySize;

    /**
     * Creates and allocates a new array.
     *
     * @param stride The stride size of the array.
     * @param batchSize The number of strides in the array.
     * @param size The number of elements in the array.
     * @param subArraySize The size of the sub arrays.
     */
    public DStrideArray1d(int stride, int batchSize, int size, int subArraySize) {
        super(size);
        this.stride = stride;
        this.batchSize = batchSize;
        this.subArraySize = subArraySize;
    }

    /**
     * Constructs aStride array from an array.
     *
     * @param src the underlying array.
     * @param stride The Size of each stride.
     * @param batchSize The number of strides.
     * * @param subArraySize The size of the sub arrays.
     * @param subArraySize The number of elements in each sub array.
     */
    public DStrideArray1d(DArray src, int stride, int batchSize, int subArraySize) {
        super(src, 0, src.size(), src.ld());
        this.stride = stride;
        this.batchSize = batchSize;
        this.subArraySize = subArraySize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray1d set(Handle handle, DArray from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray1d set(Handle handle, DArray1d src) {
        super.set(handle, src); 
        return this;
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray1d copy(Handle handle) {
        return new DStrideArray1d(stride, batchSize, size, 0).set(handle, this);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray2d as2d(int entriesPerLine) {
        return new DArray2d(this, entriesPerLine);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray3d as3d(int entriesPerLine, int linesPerLayer) {
        return new DArray3d(this, entriesPerLine, linesPerLayer);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d sub(int start, int length) {
        return new DArray1d(this, start, length);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int strideLines() {
        return stride;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int batchSize() {
        return batchSize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int subArraySize() {
        return subArraySize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray2d as2d(int entriesPerLine, int ld) {
        return new DArray2d(this, entriesPerLine, ld);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        return new DArray3d(this, entriesPerLine, ld, linesPerLayer);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d getGrid(int arrayIndex) {
        return new DArray1d(this, arrayIndex * strideLines(), arrayIndex * strideLines() + subArraySize());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d sub(int start, int size, int ld) {
        return new DArray1d(this, start, size, ld);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int size() {
        return super.size() * batchSize();
    }

}
