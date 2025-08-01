package imageWork.vectors;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Disk;
import MathSupport.Interval;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import imageWork.MyImageStack;
import imageWork.Pencil;
import imageWork.VecManager2d;
import java.awt.Color;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;

/**
 * This class extends {@link Dimensions} and provides functionality to create an
 * {@link ImagePlus} from vector and intensity data stored in
 * {@link FStrideArray3d}. It processes 3D vector fields and corresponding
 * intensities to generate an image stack displaying the vector field in 3d,
 * with spacing between each of the vectors.
 *
 * @author E. Dov Neimand
 */
public abstract class VectorImg implements Pencil{

    protected final ImageProcessor[][] processor;
    protected final Dimensions targetSpace;
    private final int spacingXY, spacingZ, r;
    private final P2dToF2d vecs, intensity;
    private final double tolerance;
    protected final Dimensions dim;
    private final Handle handle;

    /**
     * The dimensions for the output vector space.
     *
     * @param src The input vector space.
     * @param spacingXY How much space will there be between vectors in the xy
     * plane.
     * @param spacingZ How much space will there be between vectors in the z
     * dimension.
     * @param vecMag The length of the vectors.
     * @param matchHW Leave this null, unless you want the output space to match
     * the height and width in these dimensions.
     * @return The output vector space.
     */
    public static Dimensions space(Dimensions src, int spacingXY, int spacingZ, int vecMag, Dimensions matchHW) {
        return new Dimensions(null,
                matchHW == null ? (src.height - 1) * spacingXY + vecMag + 2 : matchHW.height,
                matchHW == null ? (src.width - 1) * spacingXY + vecMag + 2 : matchHW.width,
                src.hasDepth() ? (src.depth - 1) * spacingZ + vecMag + 3 : 1,
                src.batchSize);
    }

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
    public VectorImg(Dimensions overlay, Handle handle, int vecMag, P2dToF2d vecs, P2dToF2d intensity, int spacingXY, int spacingZ, double tolerance) {
        this.handle = handle;
        this.dim = new Dimensions(intensity);

        targetSpace = space(dim, spacingXY, spacingZ, vecMag, overlay);

        processor = new ImageProcessor[targetSpace.batchSize][targetSpace.depth];
        for (int t = 0; t < dim.batchSize; t++) {
            Arrays.setAll(processor[t], z -> initProcessor());
        }

        r = vecMag / 2;
        this.vecs = vecs;
        this.intensity = intensity;
        this.spacingXY = spacingXY;
        this.spacingZ = spacingZ;
        this.tolerance = tolerance;
    }

    /**
     * returns a new instance of the processor to be used in this class. The new
     * processro should be constructed with targetSpace.width and
     * targetSpace.height.
     *
     * @return A processor.
     */
    protected abstract ImageProcessor initProcessor();

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
     * @param es For multi threading.
     * @return The generated {@link ImagePlus}.
     */
    public ImageStack imgStack(ExecutorService es) {

        for (int t = 0; t < dim.batchSize; t++) {
            for (int z = 0; z < dim.depth; z++) {

                float[] currentIntensitySlice = new float[dim.layerSize()];

                intensity.get(z, t).getVal(handle).get(handle, currentIntensitySlice);

//                drawLayer(currentIntensitySlice, new VecManager2d(dim).setFrom(vecs, t, z, handle), t, z).run();
                es.submit(drawLayer(currentIntensitySlice, new VecManager2d(dim).setFrom(vecs, t, z, handle), t, z));
            }
        }

        return getStack();
    }

    /**
     * Draws all the vectors on the given layer of the frame.
     *
     * @param currentIntensitySlice The intensity data for the layer to be
     * drawn.
     * @param gridvecs The vectors to be drawn.
     * @param t The index of the frame to be drawn. This is so that the vectors
     * get drawn in the correct place.
     * @param z The depth of the layer to be drawn.
     * @return The lambda expression to draw the layer.
     */
    private Runnable drawLayer(float[] currentIntensitySlice, VecManager2d gridVecs, int t, int z) {

        return () -> {

            Interval line = new Interval();
            Disk disk = new Disk();
            Point3d vec = new Point3d(), holder[] = new Point3d[3], loc = new Point3d().setZ(z);
            Arrays.setAll(holder, i -> new Point3d());

            Pencil drawer = this;
            int[] colorHolder = new int[4];

            for (; loc.xI() < dim.width; loc.incX()) {

                int colIndex = loc.xI() * dim.height;

                for (loc.setY(0); loc.yI() < dim.height; loc.incY()) {
                    double coh = currentIntensitySlice[colIndex + loc.yI()];
                    if (Math.abs(coh) > tolerance) {
                        gridVecs.get(loc.yI(), loc.xI(), vec);

                        if (coh > tolerance)
                            buildAndDrawVec(line, vec, holder[0], loc, t, drawer, colorHolder);
                        else if (coh < -tolerance)
                            disk.set(loc, vec, r).draw(drawer, holder[0], holder[1], holder[2], t);

                    }
                }

            }
        };
    }

    /**
     * Builds and draws a line segment from the vector, then draws it.
     *
     * @param line A place holder for the line to be built.
     * @param vec The vector that defines the orientation of the line.
     * @param delta A place holder used for drawing the line.
     * @param loc The location of the center of the line.
     * @param t The frame index.
     * @param drawer The tool the line will be drawn with.
     */
    private void buildAndDrawVec(Interval line, Point3d vec, Point3d delta, Point3d loc, int t, Pencil drawer, int[] colorHolder) {
        if (vec.isFinite()) {
            drawer.setColor(color(vec, colorHolder));

            line.getA().set(loc.x() * spacingXY, loc.y() * spacingXY, loc.z() * spacingZ).translate(r + 1, r + 1, dim.depth == 1 ? 0 : r + 1);
            line.getB().set(line.getA());
            line.getA().translate(vec.scale(r));
            line.getB().translate(vec.scale(-1));

            line.draw(drawer, vec, delta, t);

        }
    }

    /**
     * Creates the image stack from the processors.
     *
     * @return The image stack.
     */
    private MyImageStack getStack() {
        MyImageStack stack = new MyImageStack(targetSpace.width, targetSpace.height);
        for (ImageProcessor[] frame : processor)
            for (ImageProcessor slice : frame)
                stack.addSlice(slice);
        
        return stack;
    }

    /**
     * Computes and saves the color based on the given vector's components. This
     * vector is expected to be normalized.
     *
     * @param vec The normalized vector.
     */
    /**
     * {@inheritDoc }
     */
    public abstract int[] color(Point3d vec, int[] colorGoesHere);

    private int[] color;

    /**
     * {@inheritDoc }
     */
    @Override
    public void setColor(Color color) {
        this.color = new int[]{
            color.getAlpha(),
            color.getRed(),
            color.getGreen(),
            color.getBlue()
        };
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void setColor(int[] color) {
        this.color = color;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void mark(Point3d p, int t) {
        processor[t][p.zI()].putPixel(p.xI(), p.yI(), color);
    }

}
