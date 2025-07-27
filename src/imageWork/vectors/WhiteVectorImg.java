package imageWork.vectors;

import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.process.BinaryProcessor;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;
import java.util.Arrays;

/**
 * Creates a vector field where all vectors are white.
 *
 * @author E. Dov Neimand
 */
public class WhiteVectorImg extends VectorImg {

    /**
     * Constructs a new VectorImg with the specified parameters to generate an
     * ImagePlus displaying the vector field from vector and intensity data.
     *
     * @param overlay The full dimensions without down sampling. Leave this null
     * if overlay is false.
     * @param handle The context
     * @param vecMag The magnitude of the vectors.
     * @param vecs The {@link FStrideArray3d} containing vector data.
     * @param intensity The {@link FStrideArray3d} containing intensity data.
     * Pass null to set all intensities to one.
     * @param spacingXY The spacing between pixels in the xy plane.
     * @param spacingZ The spacing between vectors in the z dimension.
     * @param tolerance If useNon0Intensities is false then this determines the
     * threshold for what is close to 0.
     */
    public WhiteVectorImg(Dimensions overlay, Handle handle, int vecMag, P2dToF2d vecs, P2dToF2d intensity, int spacingXY, int spacingZ, double tolerance) {
        super(overlay, handle, vecMag, vecs, intensity, spacingXY, spacingZ, tolerance);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    protected ImageProcessor initProcessor() {
        return new BinaryProcessor(new ByteProcessor(targetSpace.width, targetSpace.height));
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public int[] color(Point3d vec, int[] colorGoesHere) {
        Arrays.fill(colorGoesHere, 255);
        return colorGoesHere;
    }

}
