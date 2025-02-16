package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * This class provides functionalities to create and manipulate double arrays on
 * the GPU.
 *
 * When this class is used to store tensor data, we assume that each row of this
 * 3d matrix contains exactly one column of the tensor.
 *
 * For more methods that might be useful here, see:
 * https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
 *
 * TODO: create arrays other than double.
 *
 * @author E. Dov Neimand
 */
public class DArray3d extends Array3d implements DLineArray {

    /**
     * Creates an array with garbage values.
     *
     * @param entriesPerLine The number of entries on each line.
     * @param linesPerLayer THe number of lines in each layer.
     * @param depth The number of layers.
     */
    public DArray3d(int entriesPerLine, int linesPerLayer, int depth) {
        super(entriesPerLine, linesPerLayer, depth, Sizeof.DOUBLE);
    }

    /**
     * Constructs a sub array.
     *
     * @param src The array this one is to be a sub array of.
     * @param startLine The index of first line on each layer to start this 3d
     * array.
     * @param numLines The number of lines on each layer.
     * @param startEntry The index of the first entry on each line.
     * @param numEntryPerLine The number of entries on each line.
     * @param startLayer The index of the first layer.
     * @param numLayers The number of layers.
     */
    public DArray3d(LineArray src, int startEntry, int numEntryPerLine, int startLine, int numLines, int startLayer, int numLayers) {
        super(src, startLine, numLines, startEntry, numEntryPerLine, startLayer, numLayers);
    }

    /**
     * {@inheritDoc }
     */
    public DArray3d sub(int startEntry, int numEntryPerLine, int startLine, int numLines, int startLayer, int numLayers) {
        return new DArray3d(this, startEntry, numEntryPerLine, startLine, numLines, startLayer, numLayers);
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param linesPerLayer The number of lines in each layer.
     */
    public DArray3d(Array src, int entriesPerLine, int linesPerLayer) {
        super(src, entriesPerLine, linesPerLayer);
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param ld The number of elements that could be fit between the first
     * element of each line.
     * @param linesPerLayer The number of lines in each layer.
     */
    public DArray3d(Array src, int entriesPerLine, int ld, int linesPerLayer) {
        super(src, entriesPerLine, ld, linesPerLayer);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {

        StringBuilder sb = new StringBuilder();
        
        for(int i = 0; i < layersPerGrid(); i++)
            sb.append(getLayer(i).toString()).append("\n");
        
        return sb.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray3d set(Handle handle, DArray from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray3d copy(Handle handle) {
        return new DArray3d(linesPerLayer(), entriesPerLine(), layersPerGrid)
                .set(handle, this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray1d depth(int entryIndex, int lineIndex) {
        return new DArray1d(this, ld() * lineIndex + entryIndex, layersPerGrid(), ld() * linesPerLayer());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray3d as3d(int linesPerLayer) {
        return this;
    }
    
    /**
     * returns the desired layer of this array.
     * @param layerIndex The index of the desired layer.
     * @return The layer at the given index.
     */
    public DArray2d getLayer(int layerIndex){
        return new DArray2d(this, 0, entriesPerLine(), layerIndex * linesPerLayer(), linesPerLayer());
    }

    
    /**
     * {@inheritDoc }
     */
    @Override
    public Singleton get(int index) {
        return new DSingleton(this, index);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public double[] get(Handle handle) {
        double[] cpuArray = new double[size()];
        get(handle, Pointer.to(cpuArray));
        return cpuArray;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray3d set(Handle handle, double... srcCPUArray) {
        DLineArray.super.set(handle, srcCPUArray); 
        return this;
    }
    
    
}
