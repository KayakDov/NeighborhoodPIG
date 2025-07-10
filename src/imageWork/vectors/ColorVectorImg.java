package imageWork.vectors;

import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

/**
 * Creates color vector fields.
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
     * {@inheritDoc }
     */
    @Override
    protected ImageProcessor initProcessor() {
        return new ColorProcessor(targetSpace.width, targetSpace.height);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Pencil getPencil() {
        return new ColorPencil();
    }

    /**
     * A color pencil for creating color images.
     */
    public class ColorPencil implements Pencil {

        private int color;

        /**
         * {@inheritDoc }
         */
        @Override
        public void setColor(Point3d vec) {

            System.out.println("imageWork.VectorImg.Pencil.setColor() vec is " + vec.toString());

            color = ((int) Math.round(((vec.x() + 1) / 2.0 * 255.0)) << 16)
                    | ((int) Math.round((vec.y() * 255.0)) << 8)
                    | ((int) Math.round(((vec.z() + 1) / 2.0 * 255.0)));
        }

        /**
         * {@inheritDoc }
         */
        public void mark(Point3d p, int t) {
            processor[t][p.zI()].putPixel(p.xI(), p.yI(), color);
        }
    }

}
