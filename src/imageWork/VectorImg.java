package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Cube;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import java.awt.Dimension;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * This class extends {@link Dimensions} and provides functionality to create an
 * {@link ImagePlus} from vector and intensity data stored in
 * {@link FStrideArray3d}. It processes 3D vector fields and corresponding
 * intensities to generate an image stack where each slice represents a layer of
 * the 3D volume.
 *
 * @author E. Dov Neimand
 */
public class VectorImg extends Dimensions {

    /**
     * Inner class to manage vector data.
     */
    private class VecManager {

        private float[] vecs;

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
            gpuStrideArray.getSubArray(gridIndex).get(handle, vecs);
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

    /**
     * Constructs a new VectorImg with the specified handle and dimensions.
     *
     * @param dims The dimensions of the image.
     */
    public VectorImg(Dimensions dims) {
        super(dims);
    }

    /**
     * Creates an {@link ImagePlus} from vector and intensity data.
     *
     * @param vecs The {@link FStrideArray3d} containing vector data.
     * @param intensity The {@link FStrideArray3d} containing intensity data.
     * @param spacing The spacing between pixels.
     * @param vecMag The magnitude of the vectors.
     * @param sourceFileNames An array of source file names for each layer.
     * @return The generated {@link ImagePlus}.
     */
    public ImagePlus get(FStrideArray3d vecs, FStrideArray3d intensity, int spacing, int vecMag, String[] sourceFileNames) {

        Cube space = new Cube((width - 1) * spacing + vecMag, (height - 1) * spacing + vecMag, (depth - 1) * spacing + vecMag);

        ImageStack stack = new ImageStack(space.width(), space.height());
        FloatProcessor[] fp = new FloatProcessor[space.depth() + 1];

        Arrays.setAll(fp, i -> new FloatProcessor(space.width(), space.height()));

        float[] gridIntensity = new float[tensorSize()];
        VecManager gridVecs = new VecManager(tensorSize() * 3);
        float[] vec = new float[3];

        IntStream.range(0, batchSize).parallel().forEach( t -> {

            gridVecs.setFrom(vecs, t);
            intensity.getSubArray(t).get(handle, gridIntensity);

            int r = vecMag / 2;

            IntStream.range(0, depth).parallel().forEach(layer -> {
                for (int col = 0; col < width; col++)
                    for (int row = 0; row < height; row++)
                        for (int mag = -r; mag < r; mag++) {
                            gridVecs.get(row, col, layer, vec);
                            int x = r + spacing * col + Math.round(mag * vec[0]),
                                    y = r + spacing * row + Math.round(mag * vec[1]),
                                    z = r + spacing * layer + Math.round(mag * vec[2]);
                            fp[z].setf(x, y, gridIntensity[layer * layerSize() + col * height + row]);

                        }
            });
        });
        

        Arrays.stream(fp).forEach(ip -> stack.addSlice(ip));

        ImagePlus image = new ImagePlus("Nematics", stack);
        image.setDimensions(1, depth, batchSize);
        return image;
    }

}
