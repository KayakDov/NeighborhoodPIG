package fijiPlugin;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3StrideDim;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DStrideArray3d;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient extends TensorOrd3StrideDim implements AutoCloseable {


    private DStrideArray3d x, y, z;

    /**
     * Compute gradients of an image in both the x and y directions. Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     *
     */
    public Gradient(Handle handle, DStrideArray3d pic, Handle hand) {
        super(handle, pic);

        x = pic.copyDim();
        y = pic.copyDim();
        z = pic.copyDim();

        try (IArray dim = new IArray(handle, height, width, depth, batchSize, layerDist, tensorSize(), tensorSize() * batchSize)) {
            
            Kernel.run("batchGradients", hand,
                    x.size()*3,
                    pic,
                    P.to(dim),
                    P.to(x),P.to(x),P.to(z),
            );
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
        data.close();
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

    @Override
    public Array3d array() {
        return data;
    }
    

}
