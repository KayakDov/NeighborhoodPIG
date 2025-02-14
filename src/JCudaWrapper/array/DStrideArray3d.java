package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class DStrideArray3d extends DArray3d implements StrideArray{

    public final int stride, batchSize;

    /**
     * Constructs an array for disjoint adjacent order-3-tensors in memory.
     * @param entriesPerLine The number of elements in each line.
     * @param linesPerLayer The number of lines in each layer.
     * @param layersPerGrid The number of layers in each grid.
     * @param numGrids The number of tensors.
     */
    public DStrideArray3d(int entriesPerLine, int linesPerLayer, int layersPerGrid, int numGrids){
        super(entriesPerLine, linesPerLayer, layersPerGrid);
        this.stride = ld()*linesPerLayer*layersPerGrid;
        this.batchSize = numGrids;
    }
    
    /**
     * Constructs this array from a 1d array.
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param linesPerLayer The number of lines in each layer.
     * @param layersPerGrid The number of layers in a grid.
     * @param numGrids The number of grids in the batch.
     */
    public DStrideArray3d(DArray src, int entriesPerLine, int linesPerLayer, int layersPerGrid, int numGrids){
        super(src, entriesPerLine, linesPerLayer, layersPerGrid*numGrids);
        this.stride = ld()*linesPerLayer*layersPerGrid;
        this.batchSize = numGrids;
    }

    /**
     * An empty array with all the same dimensions as this one.
     * @return An empty array with all the same dimensions as this one.
     */
    public DStrideArray3d copyDim(){
        return new DStrideArray3d(entriesPerLine(), linesPerLayer(), numLayers(), batchSize());
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d as1d() {
        return new DArray1d(this, 0, size());
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    public int size() {
        return super.size()*batchSize();
    }
    
    

    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray3d set(Handle handle, Pointer fromCPU) {
        super.set(handle, fromCPU);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray3d set(Handle handle, DArray from) {
        super.set(handle, from);
        return this;
    }

    
    /**
     * {@inheritDoc }
     */
    @Override
    public DStrideArray3d copy(Handle handle) {
        return new DStrideArray3d(entriesPerLine(), linesPerLayer(), numLayers(), batchSize)
                .set(handle, this);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int stride() {
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
        return ld()*linesPerLayer()*numLayers()*batchSize();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray3d getSubArray(int arrayIndex) {
        return new DArray3d(
                this, 
                0, entriesPerLine(), 0, linesPerLayer(), 
                stride()*arrayIndex, 
                stride()*arrayIndex + subArraySize()
        );
    }

    /**
     * The number of layers in each sub array.
     * @return The number of layers in each sub array.
     */
    @Override
    public int numLayers() {
        return super.numLayers()/batchSize();
    }
    
    
}
