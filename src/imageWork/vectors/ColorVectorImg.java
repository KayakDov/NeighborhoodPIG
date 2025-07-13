package imageWork.vectors;

import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;

/**
 * Creates color vector fields.
 *
 * @author E. Dov Neimand
 */
public class ColorVectorImg extends VectorImg {

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
    public ColorVectorImg(Dimensions overlay, Handle handle, int vecMag, PArray2dToF2d vecs, PArray2dToF2d intensity, int spacingXY, int spacingZ, double tolerance) {
        super(overlay, handle, vecMag, vecs, intensity, spacingXY, spacingZ, tolerance);
    }

    /**
     * clamps the double between 0 and 255.
     * @param v the value to be clamped.
     * @return The closest int to v that is between 0 and 255.
     */
    private int colorClamp(double v) {
        return Math.max(0, Math.min(255, (int) Math.round(v)));
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    protected ImageProcessor initProcessor() {
        return new ColorProcessor(targetSpace.width, targetSpace.height);
    }

    public int color(Point3d vec) {

        int r = colorClamp((int) Math.round(((vec.x() + 1) / 2.0 * 255.0))),
                g = colorClamp((int) Math.round((vec.y() * 255.0))),
                b = colorClamp((int) Math.round(((vec.z() + 1) / 2.0 * 255.0)));

        return dim.hasDepth()
                ? (r << 16)
                | (g << 8)
                | b
                : Color.HSBtoRGB((float) ((Math.atan2(vec.y(), vec.x()) + Math.PI / 2) / Math.PI), 1.0f, 1.0f);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void mark(Point3d p, int t, int color) {
        processor[t][p.zI()].set(p.xI(), p.yI(), color);
    }
}
