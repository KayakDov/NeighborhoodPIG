package fijiPlugin;

import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Int.IArray1d;
import java.util.Arrays;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient extends Dimensions implements AutoCloseable {

    /**
     * 0 is the x dimension, 1 is the y dimension, and 2 is the z dimension.
     */
    public final FStrideArray3d[] x;

    /**
     * Compute gradients of an image in both the x and y directions. Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param handle The context
     * @param pic The pixel intensity values matrix.
     *
     *
     */
    public Gradient(Handle handle, FStrideArray3d pic) {
        super(handle, pic);
        x = new FStrideArray3d[3];
        Arrays.setAll(x, i -> pic.copyDim());

        //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
        int[] dimensions = new int[]{
            height, //0 -> height
            width, //1 -> width
            depth, //2 -> depth
            batchSize,//3 -> numTensorts
            height * width,//4 -> layerSize
            tensorSize(),//5  -> tensorSize 
            tensorSize() * batchSize,//6 -> batchSize (number of elements, not tensors, in the batch)
            pic.ld() //7 ld
        };

        try (IArray1d dim = new IArray1d(8).set(handle, dimensions)) {

            Kernel.run("batchGradients", handle,
                    pic.size() * 3,
                    pic,
                    P.to(dim),
                    P.to(x[0]), P.to(x[0].ld()),
                    P.to(x[1]), P.to(x[1].ld()),
                    P.to(x[2]), P.to(x[2].ld())
            );            
        }
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        for (FStrideArray3d grad : x) grad.close();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        return super.toString() + "\nd\\grad =\n" + Arrays.toString(x);
    }

    /**
     * An empty array with the same dimensions as one of the gradients.
     * @return An empty array with the same dimensions as one of the gradients.
     */
    public FStrideArray3d copyDim() {
        return x[0].copyDim();
    }

}
