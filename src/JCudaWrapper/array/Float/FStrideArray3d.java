package JCudaWrapper.array.Float;

import JCudaWrapper.array.LineArray;
import JCudaWrapper.array.StrideArray;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 *
 * @author E. Dov Neimand
 */
public class FStrideArray3d extends FArray3d implements StrideArray {

    public final int strideLines, batchSize;

    /**
     * Constructs an array for disjoint adjacent order-3-tensors in memory.
     *
     * @param entriesPerLine The number of elements in each line.
     * @param linesPerLayer The number of lines in each layer.
     * @param layersPerGrid The number of layers in each grid.
     * @param numGrids The number of tensors.
     */
    public FStrideArray3d(int entriesPerLine, int linesPerLayer, int layersPerGrid, int numGrids) {
        super(entriesPerLine, linesPerLayer, layersPerGrid * numGrids);
        this.strideLines = linesPerLayer * layersPerGrid;
        this.batchSize = numGrids;
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param startEntry The index of the first entry within each line.
     * @param entriesPerLine The number of entries in each line.
     * @param numGrids The number of grids in the batch.
     * @param startGrid The index of the first grid.
     */
    public FStrideArray3d(LineArray src, int startEntry, int entriesPerLine, int startGrid, int numGrids) {
        super(
                src,
                startEntry, entriesPerLine,
                startGrid * src.layersPerGrid(), numGrids * src.layersPerGrid()
        );

        this.strideLines = ld() * src.linesPerLayer() * src.layersPerGrid();
        this.batchSize = numGrids;
    }

    /**
     * An empty array with all the same dimensions as this one.
     *
     * @return An empty array with all the same dimensions as this one.
     */
    public FStrideArray3d copyDim() {
        return new FStrideArray3d(entriesPerLine(), linesPerLayer(), layersPerGrid(), batchSize());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray1d as1d() {
        return new FArray1d(this, 0, size());
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int size() {
        return super.size() * batchSize();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray3d set(Handle handle, Pointer fromCPU) {
        super.set(handle, fromCPU);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray3d set(Handle handle, FArray from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray3d copy(Handle handle) {
        return new FStrideArray3d(entriesPerLine(), linesPerLayer(), layersPerGrid(), batchSize)
                .set(handle, this);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int strideLines() {
        return strideLines;
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
        return entriesPerLine() * linesPerLayer() * layersPerGrid();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray3d getGrid(int gridIndex) {

        return new FArray3d(
                this,
                0, entriesPerLine(),
                layersPerGrid() * gridIndex,
                layersPerGrid()
        );
    }

    /**
     * The number of layers in each sub array.
     *
     * @return The number of layers in each sub array.
     */
    @Override
    public int layersPerGrid() {
        return super.layersPerGrid() / batchSize();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < batchSize(); i++)
            sb.append(getGrid(i).toString()).append("\n");

        return sb.toString();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FStrideArray3d set(Handle handle, float... srcCPUArray) {
        super.set(handle, srcCPUArray);
        return this;
    }

}
