package fijiPlugin;

import JCudaWrapper.array.DStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;

/**
 * Represents the dimensions and strides of a 3D tensor with additional stride
 * and batch size information.
 *
 * This class provides methods to calculate the size of individual layers, the
 * full tensor, and the total size across multiple batches.
 *
 * <p>
 * The parameters include the tensor's dimensions, distances between elements
 * (strides), and batch size, which are essential for handling tensors in
 * CUDA-based operations.</p>
 *
 * @author E.Dov Neimand
 */
public abstract class Dimensions implements AutoCloseable{

    /**
     * The height (number of rows) of the tensor.
     */
    public final int height;

    /**
     * The width (number of columns) of the tensor.
     */
    public final int width;

    /**
     * The depth (number of layers) of the tensor.
     */
    public final int depth;

    /**
     * The stride size (distance between elements in memory).
     */
    public final int strideSize;

    /**
     * The batch size (number of tensors in a batch).
     */
    public final int batchSize;
    
    public final Handle handle;

    /**
     * Constructs a new TensorOrd3dStrideDim with the specified dimensions,
     * strides, and batch size.
     *
     * @param handle The handle.
     * @param height The height (number of rows) of the tensor.
     * @param width The width (number of columns) of the tensor.
     * @param depth The depth (number of layers) of the tensor.
     * @param layerDist The distance between consecutive layers in memory.
     * @param strideSize The stride size (distance between elements in memory).
     * @param batchSize The number of tensors in a batch.
     */
    public Dimensions(Handle handle, int height, int width, int depth, int layerDist, int strideSize, int batchSize) {
        this.handle = handle;
        this.height = height;
        this.width = width;
        this.depth = depth;
        this.strideSize = strideSize;
        this.batchSize = batchSize;

        if ((long) width * height * depth * batchSize > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Image size exceeds array limit.");

    }
    
    /**
     * Constructs a new TensorOrd3dStrideDim with the specified dimensions,
     * strides, and batch size.
     *
     * @param handle The handle.
     * @param height The height (number of rows) of the tensor.
     * @param width The width (number of columns) of the tensor.
     * @param depth The depth (number of layers) of the tensor.
     * @param batchSize The number of tensors in a batch.
     */
    public Dimensions(Handle handle, int height, int width, int depth, int batchSize) {
        this(handle, height, width, depth, height*width, height*width*depth, batchSize);
    }

    
    /**
     * Copy constructor.
     *
     * @param handle The context.
     * @param copyFrom The item being copied.
     */
    public Dimensions(Handle handle, DStrideArray3d copyFrom) {
        this(handle, copyFrom.entriesPerLine(), copyFrom.linesPerLayer(), copyFrom.layersPerGrid(), copyFrom.batchSize);
    }
        
    /**
     * Copy constructor.
     *
     * @param handle The context.
     * @param copyFrom The item being copied.
     */
    public Dimensions(Handle handle, Dimensions copyFrom) {
        this(handle, copyFrom.height, copyFrom.width, copyFrom.depth, copyFrom.batchSize);
    }
    
    /**
     * Calculates the size of a single layer (width × height).
     *
     * @return The number of elements in a single layer.
     */
    public int layerSize() {
        return width * height;
    }

    /**
     * Calculates the size of the entire tensor (layer size × depth).
     *
     * @return The number of elements in the tensor.
     */
    public int tensorSize() {
        return layerSize() * depth;
    }

    /**
     * Calculates the total size of the tensor across all batches (tensor size ×
     * batch size).
     *
     * @return The total number of elements across all batches.
     */
    public int size() {
        return tensorSize() * batchSize;
    }

    @Override
    public String toString() {
        return "TensorOrd3dStrideDim {"
                + "height =" + height
                + ", width =" + width
                + ", depth =" + depth
                + ", strideSize =" + strideSize
                + ", batchSize =" + batchSize
                + ", layerSize =" + layerSize()
                + ", tensorSize =" + tensorSize()
                + ", totalSize= " + size()
                + '}';
    }
    
    /**
     * An empty array with these dimensions.
     * @return An empty array with these dimensions.
     */
    public DStrideArray3d empty(){
        return new DStrideArray3d(height, width, depth, batchSize);
    }

}