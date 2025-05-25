package JCudaWrapper.array.Float;

import JCudaWrapper.array.StrideArray;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class FStrideArray1d extends FArray1d implements StrideArray {

    public final int stride, batchSize, subArraySize;
    
    
    /**
     * Creates and allocates a new array.
     *
     * @param stride The stride size of the array.
     * @param batchSize The number of strides in the array.
     * @param size The number of elements in the array.
     * @param subArraySize The size of the sub arrays.
     */
    public FStrideArray1d(int stride, int batchSize, int size, int subArraySize) {
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
    public FStrideArray1d(FArray src, int stride, int batchSize, int subArraySize) {
        super(src, 0, src.size(), src.ld());
        this.stride = stride;
        this.batchSize = batchSize;
        this.subArraySize = subArraySize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray1d set(Handle handle, FArray from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray1d set(Handle handle, FArray1d src) {
        super.set(handle, src); 
        return this;
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray1d copy(Handle handle) {
        return new FStrideArray1d(stride, batchSize, size, 0).set(handle, this);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray2d as2d(int entriesPerLine) {
        return new FArray2d(this, entriesPerLine);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray3d as3d(int entriesPerLine, int linesPerLayer) {
        return new FArray3d(this, entriesPerLine, linesPerLayer);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray1d sub(int start, int length) {
        return new FArray1d(this, start, length, ld());
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
    public FArray2d as2d(int entriesPerLine, int ld) {
        return new FArray2d(this, entriesPerLine, ld);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        return new FArray3d(this, entriesPerLine, ld, linesPerLayer);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray1d getGrid(int arrayIndex) {
        return new FArray1d(this, arrayIndex * strideLines(), arrayIndex * strideLines() + subArraySize(), ld());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray1d sub(int start, int size, int ld) {
        return new FArray1d(this, start, size, ld);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int size() {
        return super.size() * batchSize();
    }


}
