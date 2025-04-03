package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Cube;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
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
public class VectorImg extends Dimensions {

    /**
     * Inner class to manage vector data.
     */
    private class VecManager {

        private final float[] vecs;

        /**
         * Constructs a new VecManager with the specified size.
         *
         * @param size The size of the vector data array.
         */
        public VecManager(int size) {
            this.vecs = new float[size];
        }

        /**
         * Sets the vector data from the given {@link FStrideArray3d} at the
         * specified grid index.
         *
         * @param gpuStrideArray The {@link FStrideArray3d} containing the
         * vector data.
         * @param gridIndex The index of the grid to retrieve data from.
         */
        public void setFrom(FStrideArray3d gpuStrideArray, int gridIndex) {
            gpuStrideArray.getGrid(gridIndex).get(handle, vecs);
        }

        /**
         * Calculates the index of a vector in the vector data array.
         *
         * @param row The row index.
         * @param col The column index.
         * @param layer The layer index.
         * @return The index of the vector.
         */
        private int vecIndex(int row, int col, int layer) {
            return (layer * layerSize() + col * height + row) * 3;
        }

        /**
         * Retrieves the vector at the specified row, column, and layer and
         * copies it to the provided array.
         *
         * @param row The row index.
         * @param col The column index.
         * @param layer The layer index.
         * @param vec The array to store the retrieved vector.
         */
        private void get(int row, int col, int layer, float[] vec) {
            System.arraycopy(vecs, vecIndex(row, col, layer), vec, 0, 3);
        }
    }

    private final FloatProcessor[] fp;
    private final Cube space;
    private final VecManager gridVecs;
    private final int spacing;
    private final float[] gridIntensity;
    private final float[] vec;
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
     * @param spacing The spacing between pixels.
     */
    public VectorImg(Dimensions dims, int vecMag, FStrideArray3d vecs, FStrideArray3d intensity, int spacing) {
        super(dims);

        space = new Cube(
                (width - 1) * spacing + vecMag,
                (height - 1) * spacing + vecMag,
                intensity.layersPerGrid() ==1 ? 1 : (depth - 1) * spacing + vecMag
        );

        stack = new ImageStack(space.width(), space.height());

        fp = new FloatProcessor[space.depth() + (vecs.layersPerGrid() == 1? 0:1)];

        gridIntensity = new float[tensorSize()];

        gridVecs = new VecManager(tensorSize() * 3);

        vec = new float[3];

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

        gridVecs.setFrom(vecs, t);
        intensity.getGrid(t).get(handle, gridIntensity);

        IntStream str = IntStream.range(0, depth);
        if (vecs.batchSize == 1) {
            str = str.parallel();
        }
        str.forEach(this::computeLayer);

        Arrays.stream(fp).forEach(stack::addSlice);
    }

    /**
     * Computes a layer of the image stack for a given layer index, processing
     * the vector and intensity data to set pixel values.
     *
     * @param layer The layer index.
     */
    private void computeLayer(int layer) {
        
        int layerInd = layer * layerSize();
        
        for (int col = 0; col < width; col++) {
            
            int colIndex = col*height;
            
            for (int row = 0; row < height; row++) {
                for (int mag = -r; mag < r; mag++) {
                    gridVecs.get(row, col, layer, vec);
                    int x = r + spacing * col + Math.round(mag * vec[0]),
                            y = r + spacing * row + Math.round(mag * vec[1]),
                            z = vecs.layersPerGrid() == 1? 0:r + spacing * layer + Math.round(mag * vec[2]);
                                        
                    fp[z].setf(x, y, gridIntensity[layerInd + colIndex + row]);
                }
            }
        }
    }
}
