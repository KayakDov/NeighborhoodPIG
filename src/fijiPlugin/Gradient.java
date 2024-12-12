package fijiPlugin;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.KernelManager;
import JCudaWrapper.resourceManagement.Handle;
import java.util.function.IntFunction;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient implements AutoCloseable{

    private TensorOrd3Stride dX, dY, dZ;

    /**
     * Computes the gradients of an image in both the x and y directions.
     * Gradients are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     *
     */
    public Gradient(TensorOrd3Stride pic, Handle hand) {
        
        dX = pic.emptyCopyDimensions();
        dY = pic.emptyCopyDimensions();
        dZ = pic.emptyCopyDimensions();

        KernelManager.get("batchGradients").map(hand, 3*pic.getData().length, pic.getD);
        
    }

    /**
     * An unmodifiable x gradient matrix.
     *
     * @return An unmodifiable x gradient matrix.
     */
    public TensorOrd3Stride x() {
        return dX;
    }

    /**
     * An unmodifiable y gradient matrix.
     *
     * @return An unmodifiable y gradient matrix.
     */
    public TensorOrd3Stride y() {
        return dY;
    }
        
    /**
     * An unmodifiable y gradient matrix.
     *
     * @return An unmodifiable y gradient matrix.
     */
    public TensorOrd3Stride z() {
        return dZ;
    }

    @Override
    public void close() {
        dX.close();
        dY.close();
    }

}
