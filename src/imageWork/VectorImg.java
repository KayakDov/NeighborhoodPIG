package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Interval;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.BinaryProcessor;
import ij.process.ByteProcessor;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * This class extends {@link Dimensions} and provides functionality to create an
 * {@link ImagePlus} from vector and intensity data stored in
 * {@link FStrideArray3d}. It processes 3D vector fields and corresponding
 * intensities to generate an image stack displaying the vector field in 3d,
 * with spacing between each of the vectors.
 *
 * @author E. Dov Neimand
 */
public class VectorImg {

    private final BinaryProcessor[][] processor;
    private final Dimensions targetSpace;
    private final VecManager gridVecs;
    private final int spacingXY, spacingZ, r;
    private final float[] currentIntensitySlice;
    private final PArray2dToF2d vecs, intensity;
    private final double tolerance;
    private final Dimensions dim;
    private final Handle handle;

    /**
     * The dimensions for the output vector space.
     *
     * @param src The input vector space.
     * @param spacingXY How much space will there be between vectors in the xy plane.
     * @param spacingZ How much space will there be between vectors in the z dimension.
     * @param vecMag The length of the vectors.
     * @param matchHW Leave this null, unless you want the output space to match the height and width in these dimensions.
     * @return The output vector space.
     */
    public static Dimensions space(Dimensions src, int spacingXY, int spacingZ, int vecMag, Dimensions matchHW) {
        return new Dimensions(null,
                matchHW == null? (src.height - 1) * spacingXY + vecMag + 2 : matchHW.height,
                matchHW == null ? (src.width - 1) * spacingXY + vecMag + 2: matchHW.width,
                src.hasDepth() ? (src.depth - 1) * spacingZ + vecMag + 3 : 1,
                src.batchSize);
    }

    /**
     * Constructs a new VectorImg with the specified parameters to generate an
     * ImagePlus displaying the vector field from vector and intensity data.
     *
     * @param overlay The full dimensions without down sampling.  Leave this null if overlay is false.
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
    public VectorImg(Dimensions overlay, Handle handle, int vecMag, PArray2dToF2d vecs, PArray2dToF2d intensity, int spacingXY, int spacingZ, double tolerance) {
        this.handle = handle;
        this.dim = new Dimensions(intensity);

        targetSpace = space(dim, spacingXY, spacingZ, vecMag, overlay);

        processor = new BinaryProcessor[targetSpace.batchSize][targetSpace.depth];
        for (int t = 0; t < dim.batchSize; t++)
            Arrays.setAll(processor[t], z -> new BinaryProcessor(new ByteProcessor(targetSpace.width, targetSpace.height)));

        currentIntensitySlice = new float[dim.layerSize()];

        gridVecs = new VecManager(dim);

        r = vecMag / 2;
        this.vecs = vecs;
        this.intensity = intensity;
        this.spacingXY = spacingXY;
        this.spacingZ = spacingZ;
        this.tolerance = tolerance;
    }

    /**
     * The dimensions of the output vector space.
     *
     * @return The dimensions of the output vector space.
     */
    public Dimensions getOutputDimensions() {
        return targetSpace;
    }

    /**
     * Creates an {@link ImagePlus} from the vector and intensity data provided
     * during construction.
     *
     * @return The generated {@link ImagePlus}.
     */
    public ImageStack imgStack() {

        IntStream.range(0, dim.batchSize)
                //                .parallel()  //TODO:reinstate
                .forEach(this::computeGrid);

        return getStack();
    }

    /**
     * Creates the image stack from the processors.
     *
     * @return The image stack.
     */
    private MyImageStack getStack() {
        MyImageStack stack = new MyImageStack(targetSpace.width, targetSpace.height);
        for (int t = 0; t < processor.length; t++)
            for (int z = 0; z < processor[t].length; z++)
                stack.addSlice(processor[t][z]);
        return stack;
    }

    /**
     * Computes the image of the vector field for a specific grid/frame.
     *
     * @param t The index of the desired grid.
     */
    private void computeGrid(int t) {

        IntStream str = IntStream.range(0, dim.depth);

//        if (dim.batchSize < Runtime.getRuntime().availableProcessors()) str = str.parallel();  //TODO:Reinstate
        str.forEach(z -> computeLayer(t, z));

    }

    /**
     * An object to facilitate drawing with IJ at the proffered point.
     */
    public class Pencil {

        /**
         * Draws 255 in the desired time and place.
         *
         * @param p The location to draw.
         * @param t The frame to draw on.
         */
        public void mark(Point3d p, int t) {
//            if(p.zI() > 158) System.out.println("imageWork.VectorImg.Pencil.mark() t, (x,y,z) = " + t + ", " + p);
            processor[t][p.zI()].putPixel(p.xI(), p.yI(), 255);
        }
    }

    /**
     * Computes a layer of the image stack for a given layer index, processing
     * the vector and intensity data to set pixel values.
     *
     * @param z The layer index.
     */
    private void computeLayer(int t, int z) {

        gridVecs.setFrom(vecs, t, z, handle);
        intensity.get(z, t).getVal(handle).get(handle, currentIntensitySlice);

        Interval line = new Interval();
        Point3d vec = new Point3d(), delta = new Point3d();
        Pencil drawer = new Pencil();

        for (int x = 0; x < dim.width; x++) {

            int colIndex = x * dim.height;

            for (int y = 0; y < dim.height; y++) {

                if (currentIntensitySlice[colIndex + y] > tolerance) {

                    gridVecs.get(y, x, vec);

                    if (vec.isFinite()) {
                        line.getA().set(x * spacingXY, y * spacingXY, z * spacingZ).translate(r + 1, r + 1, dim.depth == 1 ? 0 : r + 1);
                        line.getB().set(line.getA());
                        line.getA().translate(vec.scale(r));
                        line.getB().translate(vec.scale(-1));
                        
                        line.draw(drawer, vec, delta, t);
                    }
                }
            }

        }
    }

}
