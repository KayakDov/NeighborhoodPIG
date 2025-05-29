package fijiPlugin;

import FijiInput.UserInput;
import JCudaWrapper.array.Array;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import ij.ImagePlus;
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
    public final PArray2dToF2d[] x;
    public final Dimensions dim;

    /**
     * Compute gradients of an image in both the x and y directions.Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param handle The context
     * @param imp  The image from which the pixel gradients are to be taken.
     * @param ui The input from the user.
     *
     *
     */
    public Gradient(Handle handle, ImagePlus imp, UserInput ui) {        

        
        
        try (PArray2dToF2d pic = ProcessImage.processImages(handle, imp, ui)) {
            
            
            dim = new Dimensions(handle, pic.targetDim().entriesPerLine, pic.targetDim().numLines, pic.entriesPerLine(), pic.linesPerLayer());
            
            x = new PArray2dToF2d[]{dim.emptyP2dToF2d(handle), dim.emptyP2dToF2d(handle), dim.emptyP2dToF2d(handle)};

            Kernel.run("batchGradients", handle,
                    dim.size() * 3,
                    new PArray2dTo2d[]{pic, x[0], x[1], x[2]},
                    dim,
                    P.to(ui.neighborhoodSize.layerRes)
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
