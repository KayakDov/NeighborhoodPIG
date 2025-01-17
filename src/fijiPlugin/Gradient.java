package fijiPlugin;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3StrideDim;
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
public class Gradient extends TensorOrd3StrideDim implements AutoCloseable {

    private TensorOrd3Stride x, y, z;

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

        x = pic.emptyCopyDimensions();
        y = pic.emptyCopyDimensions();
        z = pic.emptyCopyDimensions();

        try (IArray dim = new IArray(handle, height, width, depth, batchSize, layerDist, tensorSize(), tensorSize() * batchSize)) {

            Kernel.run("batchGradients", hand,
                    3 * pic.dArray().length,
                    pic.dArray(),
                    P.to(dim),
                    P.to(x), P.to(y), P.to(z)
            );
        }

    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                DArray array = new DArray(hand, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 3, 3, 3, 3, 3)) {
            TensorOrd3Stride tenStr = new Matrix(hand, array, 3, 5).repeating(1);

            try (Gradient grad = new Gradient(tenStr, hand)) {
                System.out.println("dX: " + grad.x.dArray().toString());
            }
        }
    }

    /**
     * An x gradient matrix.
     *
     * @return An x gradient matrix.
     */
    public TensorOrd3Stride x() {
        return x;
    }

    /**
     * An y gradient matrix.
     *
     * @return A y gradient matrix.
     */
    public TensorOrd3Stride y() {
        return y;
    }

    /**
     * A y gradient matrix.
     *
     * @return A y gradient matrix.
     */
    public TensorOrd3Stride z() {
        return z;
    }

    @Override
    public void close() {
        x.close();
        y.close();
        z.close();
    }

    /**
     * The number of pixels for which the gradient is calculated.
     *
     * @return The number of pixels for which the gradient is calculated.
     */
    public int size() {
        return height * width * depth * batchSize;
    }

    @Override
    public String toString() {
        return super.toString() + "\nd\\dx =\n" + x.toString() + "\nd\\dy = \n" + y.toString() + "\nd\\dz = \n" + z.toString();
    }
    

}
