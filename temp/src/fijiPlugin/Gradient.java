package fijiPlugin;

import FijiInput.UsrInput;
import JCudaWrapper.array.Array;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import ij.ImagePlus;
import imageWork.MyImagePlus;
import imageWork.ProcessImage;
import java.util.Arrays;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient implements AutoCloseable {

    /**
     * 0 is the x dimension, 1 is the y dimension, and 2 is the z dimension.
     */
    public final P2dToF2d[] x;
    public final Dimensions dim;

    /**
     * Compute gradients of an image in both the x and y directions.Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param handle The context
     * @param imp The image from which the pixel gradients are to be taken.
     * @param ui The input from the user.
     *
     *
     */
    public Gradient(Handle handle, MyImagePlus imp, UsrInput ui) {

        try (P2dToF2d pic = ProcessImage.processImages(handle, imp, ui)) {
                        
            dim = imp.dim().setGpuDim(handle);
            
            x = new P2dToF2d[dim.depth > 1 ? 3 : 2];

            PArray2dTo2d[] dataParams = new PArray2dTo2d[x.length + 1];
            dataParams[0] = pic;
            for (int i = 0; i < x.length; i++) dataParams[i + 1] = x[i] = dim.emptyP2dToF2d(handle);            
            
            

                new Kernel("batchGradients" + x.length + "d").run(
                        handle,
                        dim.size() * x.length,
                        dataParams,
                        dim.getGpuDim(),
                        P.to(ui.neighborhoodSize.layerRes.orElse(1.0))
                );
            
       }

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
        return super.toString() + "\nd\\grad =\n" + Arrays.toString(x).replace("[[", "[\n[").replace(", [", ",\n[");
    }

}
