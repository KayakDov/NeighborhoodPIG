package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import MathSupport.Cube;
import MathSupport.Line;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
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
public class VectorImg extends Dimensions implements Consumer<Point3d> {

    @Override
    public void accept(Point3d t) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    private final FloatProcessor[] fp;
    private final Cube space;
    private final VecManager gridVecs;
    private final int spacing;
    private final float[] gridIntensity;
    private final int r;
    private final FStrideArray3d vecs;
    private final ImageStack stack;
    private final FStrideArray3d intensity;

    /**
     * Constructs a new VectorImg with the specified parameters to generate an
     * ImagePlus displaying the vector field from vector and intensity data.
     *
     * @param dims The dimensions of the image.
     * @param vecMag The magnitude of the vectors.
     * @param vecs The {@link FStrideArray3d} containing vector data.
     * @param intensity The {@link FStrideArray3d} containing intensity data.
     * Pass null to set all intensities to one.
     * @param spacing The spacing between pixels.
     */
    public VectorImg(Dimensions dims, int vecMag, FStrideArray3d vecs, FStrideArray3d intensity, int spacing) {
        super(dims);

        space = new Cube(
                (width - 1) * spacing + vecMag + 2,
                (height - 1) * spacing + vecMag + 2,
                vecs.layersPerGrid() == 1 ? 1 : (depth - 1) * spacing + vecMag + 2
        );

        stack = new ImageStack(space.width(), space.height());

        fp = new FloatProcessor[space.depth() + (vecs.layersPerGrid() == 1 ? 0 : 1)];

        gridIntensity = intensity == null ? null : new float[tensorSize()];

        gridVecs = new VecManager(this);

        r = vecMag / 2;
        this.vecs = vecs;
        this.intensity = intensity;
        this.spacing = spacing;

    }

    /**
     * Creates an {@link ImagePlus} from the vector and intensity data provided
     * during construction.
     *
     * @return The generated {@link ImagePlus}.
     */
    public ImagePlus get() {

        IntStream str = IntStream.range(0, batchSize);
        if (vecs.batchSize > 1) {
            str = str.parallel();
        }
        str.forEach(this::computeGrid);

        ImagePlus image = new ImagePlus("Nematics", stack);
        image.setDimensions(1, depth, batchSize);
        return image;
    }

    /**
     * Computes the image of the vector field for a specific grid/frame.
     *
     * @param t The index of the desired grid.
     */
    private void computeGrid(int t) {

        Arrays.setAll(fp, i -> new FloatProcessor(space.width(), space.height()));

        gridVecs.setFrom(vecs, t, handle);

        if (gridIntensity != null)
            intensity.getGrid(t).get(handle, gridIntensity);

        IntStream str = IntStream.range(0, depth);
        if (vecs.batchSize == 1)
            str = str.parallel();

        str.forEach(this::computeLayer);

        Arrays.stream(fp).forEach(stack::addSlice);
    }

    /**
     * An object to facilitate drawing with IJ at the proffered point.
     */
    public class Pencil implements Consumer<Point3d> {

        private float intensity = 1;

        /**
         * Sets the intensity to be applied by this pencil.
         *
         * @param intensity The shade to be drawn.
         * @return this.
         */
        public Pencil setIntensity(float intensity) {
            this.intensity = intensity;
            return this;
        }

        /**
         * {@inheritDoc }
         */
        @Override
        public void accept(Point3d p) {

            if (!p.firstQuadrant())
                throw new RuntimeException(p.toString());//TODO: delete me
            fp[p.zI()].setf(p.xI(), p.yI(), intensity);
        }
    }

    /**
     * Computes a layer of the image stack for a given layer index, processing
     * the vector and intensity data to set pixel values.
     *
     * @param layer The layer index.
     */
    private void computeLayer(int layer) {

        Line line = new Line();
        Point3d vec1 = new Point3d(), vec2 = new Point3d();
        Pencil drawer = new Pencil();

        int layerInd = layer * layerSize();
        for (int col = 0; col < width; col++) {

            int colIndex = col * height + layerInd;

            for (int row = 0; row < height; row++) {

                if (intensity != null)
                    drawer.setIntensity(gridIntensity[colIndex + row]);

                gridVecs.get(row, col, layer, vec1, r);

                if (depth == 1) vec1.setZ(0);

                line.getA().set(col, row, layer).scale(spacing).translate(r + 1, r + 1, depth == 1 ? 0 : r + 1);
                line.getB().set(line.getA());
                line.getA().translate(vec1);
                line.getB().translate(vec1.scale(-1));
                line.draw(drawer, vec1, vec2);

            }
        }

    }

}

//                
//                for (int mag = -r; mag < r; mag++) {
//                    
//                    
//                    
//                    int x = r + spacing * col + Math.round(mag * vec[0]),
//                            y = r + spacing * row + Math.round(mag * vec[1]),
//                            z = vecs.layersPerGrid() == 1 ? 0 : r + spacing * layer + Math.round(mag * vec[2]);
//
//                    fp[z].setf(x, y, gridIntensity == null ? 1 : gridIntensity[layerInd + colIndex + row]);
//                }
