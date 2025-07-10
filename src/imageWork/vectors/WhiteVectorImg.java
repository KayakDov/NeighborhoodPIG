package imageWork.vectors;

import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.process.BinaryProcessor;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

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
    public WhiteVectorImg(Dimensions overlay, Handle handle, int vecMag, PArray2dToF2d vecs, PArray2dToF2d intensity, int spacingXY, int spacingZ, double tolerance) {
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
    public Pencil getPencil() {
        return new Pencil() {
            @Override
            public void setColor(Point3d vec) {
            }

            @Override
            public void mark(Point3d p, int t) {
                processor[t][p.zI()].putPixel(p.xI(), p.yI(), 255);
            }
        };
    }
}
