package JCudaWrapper.algebra;

import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author Dov Neimand
 */
public class TensorOrd3Stride {

    private final int height, width, depth, colDist, layerDist, strideSize, batchSize;
    private final DStrideArray data;
    private Handle handle;

    /**
     * A set of order 3 tensors.
     * @param handle The handle.
     * @param height The height of the tensors.
     * @param width The width of the tensors.
     * @param depth The depth of the tensors.
     * @param colDist The distance between the first elements of the columns of the tensors.
     * @param layerDist The distance between the first elements of the layers of the tensors.
     * @param strideSize The distance between the first elements of the tensors.
     * @param batchSize The number of tensors.
     * @param data The underlying data.
     */
    public TensorOrd3Stride(Handle handle, int height, int width, int depth, int colDist, int layerDist, int strideSize, int batchSize, DStrideArray data) {
        this.height = height;
        this.width = width;
        this.depth = depth;
        this.colDist = colDist;
        this.layerDist = layerDist;
        this.strideSize = strideSize;
        this.batchSize = batchSize;
        this.data = data;
        this.handle = handle;
    }

    
    /**
     * A set of order 3 tensors.
     * @param handle The handle.
     * @param height The height of the tensors.
     * @param width The width of the tensors.
     * @param depth The depth of the tensors.
     * @param batchSize The number of tensors.
     * @param data The underlying data.
     */
    public TensorOrd3Stride(Handle handle, int height, int width, int depth, int batchSize, DStrideArray data){
        this(handle,height, width, depth, height, height*width, height*width*depth, batchSize, data);
    }
    
}
