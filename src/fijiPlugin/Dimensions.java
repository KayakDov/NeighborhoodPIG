package fijiPlugin;

import JCudaWrapper.array.Int.IArray1d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.HyperStackConverter;
import ij.process.FloatProcessor;
import java.io.Closeable;

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
public class Dimensions implements Closeable {

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
     * The batch size (number of tensors in a batch).
     */
    public final int batchSize;

    private IArray1d gpuDim;

    /**
     * Constructs a new TensorOrd3dStrideDim with the specified dimensions,
     * strides, and batch size.
     *
     * @param handle The handle. Set this to null to block creation of a
     * dimensions gpu array.
     * @param height The height (number of rows) of the tensor.
     * @param width The width (number of columns) of the tensor.
     * @param depth The depth (number of layers) of the tensor.
     * @param batchSize The number of tensors in a batch.
     */
    public Dimensions(Handle handle, int height, int width, int depth, int batchSize) {
        this.height = height;
        this.width = width;
        this.depth = depth;
        this.batchSize = batchSize;

        if ((long) width * height * depth * batchSize > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Image size exceeds array limit.");

        gpuDim = handle == null
                ? null
                : new IArray1d(handle,
                        height, //0 -> height
                        width, //1 -> width
                        depth, //2 -> depth
                        batchSize,//3 -> numTensorts
                        height * width,//4 -> layerSize
                        tensorSize(),//5  -> tensorSize 
                        tensorSize() * batchSize //6 -> batchSize (number of elements, not tensors, in the batch)
                );

    }

    /**
     * The gpu dimensions.
     *
     * @return The gpu dimensions.
     */
    public IArray1d getGpuDim() {
        return gpuDim;
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
    public final int tensorSize() {
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
                + ", batchSize =" + batchSize
                + ", layerSize =" + layerSize()
                + ", tensorSize =" + tensorSize()
                + ", totalSize= " + size()
                + '}';
    }

    /**
     * An empty array with these dimensions. The pointers in the array have been
     * allocated.
     *
     * @param hand The context
     * @return An empty array with these dimensions.
     */
    public PArray2dToD2d emptyP2dToD2d(Handle hand) {
        return new PArray2dToD2d(depth, batchSize, height, width, hand);
    }
    /**
     * An empty array with these dimensions. The pointers in the array have been
     * allocated.
     *
     * @param hand The context.  Leave this null in order to not allocate target memory.
     * @return An empty array with these dimensions.
     */
    public PArray2dToF2d emptyP2dToF2d(Handle hand) {
        return new PArray2dToF2d(depth, batchSize, height, width, hand);
    }

    /**
     * Turns the stack into a hyper stack with these dimensions.
     *
     * @param imp
     * @return
     */
    public ImagePlus setToHyperStack(ImagePlus imp) {
        if (batchSize > 1) {
            imp = HyperStackConverter.toHyperStack(
                    imp,
                    1,
                    depth,
                    batchSize
            );
        }
        return imp;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        gpuDim.close();
    }

    /**
     * An image stack with this width and height.
     *
     * @return An image stack with this width and height.
     */
    public ImageStack getImageStack() {
        return new ImageStack(width, height);
    }

    /**
     * A float processor with these dimensions.
     *
     * @return A float processor with these dimensions.
     */
    public FloatProcessor getFloatProcessor() {
        return new FloatProcessor(width, height);
    }
    
    /**
     * True if depth > 1, false otherwise.
     * @return True if depth > 1, false otherwise.
     */
    public boolean hasDepth(){
        return depth > 1;
    }
    
    /**
     * 3 if depth is a dimension, otherwise 2.
     * @return 3 if depth is a dimension, otherwise 2.
     */
    public int num(){
        return hasDepth()? 3: 2;
    }
    
    
}
