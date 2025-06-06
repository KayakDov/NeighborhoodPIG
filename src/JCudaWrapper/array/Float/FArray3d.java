package JCudaWrapper.array.Float;

import JCudaWrapper.array.Double.*;
import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.LineArray;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * This class provides functionalities to create and manipulate float arrays on
 * the GPU.
 *
 * When this class is used to store tensor data, we assume that each row of this
 * 3d matrix contains exactly one column of the tensor.
 *
 * For more methods that might be useful here, see:
 * https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
 *
 * TODO: create arrays other than float.
 *
 * @author E. Dov Neimand
 */
public class FArray3d extends Array3d implements FLineArray {

    /**
     * Creates an array with garbage values.
     *
     * @param entriesPerLine The number of entries on each line.
     * @param linesPerLayer THe number of lines in each layer.
     * @param depth The number of layers.
     */
    public FArray3d(int entriesPerLine, int linesPerLayer, int depth) {
        super(entriesPerLine, linesPerLayer, depth, Sizeof.FLOAT);
    }

    /**
     * Constructs a sub array.
     *
     * @param src The array this one is to be a sub array of.
     * @param startEntry The index of the first entry on each line.
     * @param entryPerLine The number of entries on each line.
     * @param startLayer The index of the first layer.
     * @param numLayers The number of layers.
     */
    public FArray3d(LineArray src, int startEntry, int entryPerLine, int startLayer, int numLayers) {
        super(src, 
                startEntry, entryPerLine, 
                startLayer, numLayers
        );
    }

    /**
     * {@inheritDoc }
     */
    public FArray3d sub(int startEntry, int numEntryPerLine, int startLayer, int numLayers) {
        return new FArray3d(this, startEntry, numEntryPerLine, startLayer, numLayers);
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param linesPerLayer The number of lines in each layer.
     */
    public FArray3d(Array src, int entriesPerLine, int linesPerLayer) {
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
    public FArray3d(Array src, int entriesPerLine, int ld, int linesPerLayer) {
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
    public FArray3d set(Handle handle, FArray from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray3d copy(Handle handle) {
        return new FArray3d(linesPerLayer(), entriesPerLine(), layersPerGrid)
                .set(handle, this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray1d depth(int entryIndex, int lineIndex) {
        return new FArray1d(this, ld() * lineIndex + entryIndex, layersPerGrid(), linesPerLayer()*ld());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray3d as3d(int linesPerLayer) {
        return this;
    }
    
    /**
     * returns the desired layer of this array.
     * @param layerIndex The index of the desired layer.
     * @return The layer at the given index.
     */
    public FArray2d getLayer(int layerIndex){
        return new FArray2d(this, 0, entriesPerLine(), layerIndex * linesPerLayer(), linesPerLayer());
    }

    
    /**
     * {@inheritDoc }
     */
    @Override
    public Singleton get(int index) {
        return new FSingleton(this, index);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public float[] get(Handle handle) {
        float[] cpuArray = new float[size()];
        get(handle, Pointer.to(cpuArray));
        return cpuArray;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray3d set(Handle handle, float... srcCPUArray) {
        FLineArray.super.set(handle, srcCPUArray); 
        return this;
    }
    
    
}
