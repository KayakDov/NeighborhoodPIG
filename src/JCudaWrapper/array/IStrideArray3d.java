package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 *
 * @author E. Dov Neimand
 */
public class IStrideArray3d extends IArray3d implements StrideArray {

    public final int strideLines, batchSize;

    /**
     * Constructs an array for disjoint adjacent order-3-tensors in memory.
     *
     * @param entriesPerLine The number of elements in each line.
     * @param linesPerLayer The number of lines in each layer.
     * @param layersPerGrid The number of layers in each grid.
     * @param numGrids The number of tensors.
     */
    public IStrideArray3d(int entriesPerLine, int linesPerLayer, int layersPerGrid, int numGrids) {
        super(entriesPerLine, linesPerLayer, layersPerGrid * numGrids);
        this.strideLines = linesPerLayer * layersPerGrid;
        this.batchSize = numGrids;
    }


    /**
     * An empty array with all the same dimensions as this one.
     *
     * @return An empty array with all the same dimensions as this one.
     */
    public IStrideArray3d copyDim() {
        return new IStrideArray3d(entriesPerLine(), linesPerLayer(), layersPerGrid(), batchSize());
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
    public IStrideArray3d set(Handle handle, Pointer fromCPU) {
        super.set(handle, fromCPU);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    public IStrideArray3d set(Handle handle, IArray from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public IStrideArray3d copy(Handle handle) {
        return new IStrideArray3d(entriesPerLine(), linesPerLayer(), layersPerGrid(), batchSize)
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
    public DArray3d getSubArray(int arrayIndex) {
        
        return new DArray3d(
                this,
                0, entriesPerLine(), 
                0, linesPerLayer(),
                layersPerGrid() * arrayIndex,
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
            sb.append(getSubArray(i).toString()).append("\n");
        
        return sb.toString();
    }
    
}

