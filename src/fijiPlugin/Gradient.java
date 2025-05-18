package fijiPlugin;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Int.IArray1d;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Singleton;
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
    public final PArray2dToD2d[] x;

    /**
     * Compute gradients of an image in both the x and y directions.Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param handle The context
     * @param pic The pixel intensity values matrix.
     * @param layerDist Used for the distance between layers as a multiple of
     * the distance between adjacent pixels.
     *
     *
     */
    public Gradient(Handle handle, PArray2dToD2d pic, NeighborhoodDim layerDist) {
        super(handle, pic);
        x = new PArray2dToD2d[]{pic.copyDim(), pic.copyDim(), pic.copyDim()};

        Kernel.run("batchGradients", handle,
                size() * 3,
                new PArray2dTo2d[]{pic, x[0], x[1], x[2]},
                this,
                P.to(layerDist.layerRes)
        );

//        Kernel.run("batchGradients", handle, size()*3,
//                P.to(pic), P.to(pic.targetLD()), P.to(pic.targetLD().ld()), P.to(pic.ld()),
//                P.to(x[0]), P.to(x[0].targetLD()), P.to(x[0].targetLD().ld()), P.to(x[0].ld()),
//                P.to(x[1]), P.to(x[1].targetLD()), P.to(x[1].targetLD().ld()), P.to(x[1].ld()),
//                P.to(x[2]), P.to(x[2].targetLD()), P.to(x[2].targetLD().ld()), P.to(x[2].ld()),
//                P.to(gpuDim()),
//                P.to(layerDist.layerRes)
//        );

        System.out.println("fijiPlugin.Gradient.<init>() 0");
        System.out.println("fijiPlugin.Gradient.<init>() 1" + x[0].toString());
        System.out.println("fijiPlugin.Gradient.<init>() 2");

    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        for (Array grad : x) grad.close();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        return super.toString() + "\nd\\grad =\n" + Arrays.toString(x);
    }

}
