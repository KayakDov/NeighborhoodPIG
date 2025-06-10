package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Cube;
import MathSupport.Line;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.BinaryProcessor;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import java.util.Arrays;
import java.util.function.Consumer;
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

    private final BinaryProcessor[] processor;
    private final Dimensions space;
    private final VecManager gridVecs;
    private final int spacing, r;
    private final float[] gridIntensity;
    private final PArray2dToF2d vecs, intensity;
    private final ImageStack stack;
    private final double tolerance;
    private final Dimensions dim;
    private final Handle handle;

    /**
     * The dimensions for the output vector space.
     * @param src The input vector space.
     * @param spacing How much space will there be between vectors.
     * @param vecMag The length of the vectors.
     * @return The output vector space.
     */
    public static Dimensions space(Dimensions src, int spacing, int vecMag){
        return new Dimensions(null, 
                (src.height - 1) * spacing + vecMag + 2, 
                (src.width - 1) * spacing + vecMag + 2, 
                src.hasDepth() ? (src.depth - 1) * spacing + vecMag + 2 : 1, 
                src.batchSize);
    }
    
    /**
     * Constructs a new VectorImg with the specified parameters to generate an
     * ImagePlus displaying the vector field from vector and intensity data.
     *
     * @param handle The context     
     * @param vecMag The magnitude of the vectors.
     * @param vecs The {@link FStrideArray3d} containing vector data.
     * @param intensity The {@link FStrideArray3d} containing intensity data.
     * Pass null to set all intensities to one.
     * @param spacing The spacing between pixels.
     * @param tolerance If useNon0Intensities is false then this determines the
     * threshold for what is close to 0.
     */
    public VectorImg(Handle handle, int vecMag, PArray2dToF2d vecs, PArray2dToF2d intensity, int spacing, double tolerance) {
        this.handle = handle;
        this.dim = new Dimensions(intensity);
        
        space = space(dim, spacing, vecMag);

        stack = new ImageStack(space.width, space.height);

        processor = new BinaryProcessor[space.depth + (dim.hasDepth() ? 1 : 0)];

        gridIntensity = intensity == null ? null : new float[dim.tensorSize()];

        gridVecs = new VecManager(dim);

        r = vecMag / 2;
        this.vecs = vecs;
        this.intensity = intensity;
        this.spacing = spacing;
        this.tolerance = tolerance;        
    }

    /**
     * The dimensions of the output vector space.
     * @return The dimensions of the output vector space.
     */
    public Dimensions getOutputDimensions() {
        return space;
    }

    
    
    /**
     * Creates an {@link ImagePlus} from the vector and intensity data provided
     * during construction.
     *
     * @return The generated {@link ImagePlus}.
     */
    public ImageStack imgStack() {

        IntStream str = IntStream.range(0, dim.batchSize);//.parallel();
        
        str.forEach(this::computeGrid);

        return stack;
    }

    /**
     * Computes the image of the vector field for a specific grid/frame.
     *
     * @param t The index of the desired grid.
     */
    private void computeGrid(int t) {

        Arrays.setAll(processor, i -> new BinaryProcessor(new ByteProcessor(space.width, space.height)));

        IntStream str = IntStream.range(0, dim.depth);

//        if (dim.batchSize == 1) str = str.parallel();

        str.forEach(z -> computeLayer(t, z));

        Arrays.stream(processor).forEach(stack::addSlice);        
        
    }

    /**
     * An object to facilitate drawing with IJ at the proffered point.
     */
    public class Pencil implements Consumer<Point3d> {

        /**
         * {@inheritDoc }
         */
        @Override
        public void accept(Point3d p) {
            processor[p.zI()].putPixel(p.xI(), p.yI(), 255);
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
        intensity.get(z, t).getVal(handle).get(handle, gridIntensity);
                
        Line line = new Line();
        Point3d vec1 = new Point3d(), vec2 = new Point3d();
        Pencil drawer = new Pencil();
        
        for (int col = 0; col < dim.width; col++) {

            int colIndex = col * dim.height;

            for (int row = 0; row < dim.height; row++) {

                double localIntensity = gridIntensity[colIndex + row];

                if (localIntensity > tolerance) {                    

                    gridVecs.get(row, col, vec1, r);
                                        
                    if (vec1.isFinite()) {
                        if (dim.depth == 1) vec1.setZ(0);
                        
                        line.getA().set(col, row, z).scale(spacing).translate(r + 1, r + 1, dim.depth == 1 ? 0 : r + 1);
                        line.getB().set(line.getA());
                        line.getA().translate(vec1);
                        line.getB().translate(vec1.scale(-1));
        
                        //if(line.length() <= 2)System.out.println("imageWork.VectorImg.computeLayer() liune length = " + line.length() + " vec = " + vec1.toString());
                        line.draw(drawer, vec1, vec2);
                    }
                }
            }

        }
    }
    
    /**
     * Saves the vectors as a bunch of images.
     * @param parentFolder 
     */
    public void saveToFile(String parentFolder){
        new MyImagePlus("Nematic Field", imgStack(), space.depth)
                .saveSlices(parentFolder);
    }

}
