package fijiPlugin;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3dStrideDim;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient extends TensorOrd3dStrideDim implements AutoCloseable {

    private TensorOrd3Stride dX, dY, dZ;

    /**
     * Compute gradients of an image in both the x and y directions. Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     *
     */
    public Gradient(TensorOrd3Stride pic, Handle hand) {
        super(pic);

        dX = pic.emptyCopyDimensions();
        dY = pic.emptyCopyDimensions();
        dZ = pic.emptyCopyDimensions();

        try (IArray dim = new IArray(handle, height, width, depth, batchSize, layerDist, tensorSize(), tensorSize() * batchSize)) {

            Kernel.run("batchGradients", hand,
                    3 * pic.dArray().length,
                    pic.dArray(),
                    P.to(dim),
                    P.to(dX), P.to(dY), P.to(dZ)
            );
        }

    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                DArray array = new DArray(hand, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 3, 3, 3, 3, 3)) {
            TensorOrd3Stride tenStr = new Matrix(hand, array, 3, 5).repeating(1);

            try (Gradient grad = new Gradient(tenStr, hand)) {
                System.out.println("dX: " + grad.dX.dArray().toString());
            }
        }
    }

    /**
     * An x gradient matrix.
     *
     * @return An x gradient matrix.
     */
    public TensorOrd3Stride x() {
        return dX;
    }

    /**
     * An y gradient matrix.
     *
     * @return A y gradient matrix.
     */
    public TensorOrd3Stride y() {
        return dY;
    }

    /**
     * A y gradient matrix.
     *
     * @return A y gradient matrix.
     */
    public TensorOrd3Stride z() {
        return dZ;
    }

    @Override
    public void close() {
        dX.close();
        dY.close();
        dZ.close();
    }

    /**
     * The number of pixels for which the gradient is calculated.
     *
     * @return The number of pixels for which the gradient is calculated.
     */
    public int size() {
        return height * width * depth * batchSize;
    }

}
