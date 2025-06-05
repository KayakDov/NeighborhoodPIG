package imageWork;

import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import java.util.Arrays;

/**
 * A class for creating grayscale images from tensor data.
 *
 * This class processes 3D tensor data to generate grayscale intensity maps. It
 * supports GPU-accelerated processing and provides functionality for displaying
 * and saving images.
 *
 * @author E. Dov Neimand
 */
public class GrayScaleHeatMapCreator extends HeatMapCreator {

    private final PArray2dToF2d image;
    private final PArray2dToF2d coherence;
    private final double tolerance;

    /**
     * Constructs a GrayScaleImageCreator instance.
     *
     * @param sliceNames The names of the slices.
     * @param stackName The name of the stack.
     * @param handle The handle for GPU resource management.
     * @param image The tensor data used to generate the grayscale image.
     * @param coherence Values of this map are set to NaN when coherence is at
     * or very near 0. Set to null to avoid doing this, if for example you want
     * to map coherence.
     * @param tolerance Defines what is close to 0.
     * @param dim The dimensions.
     */
    public GrayScaleHeatMapCreator(String[] sliceNames, String stackName, Handle handle, PArray2dToF2d image, PArray2dToF2d coherence, double tolerance, Dimensions dim) {
        super(sliceNames, stackName, dim, handle);
        
        this.image = image;

        this.coherence = coherence;
        this.tolerance = tolerance;
    }

    /**
     * Displays the grayscale image stack in Fiji (ImageJ) as a hyperstack.
     */
    @Override
    public void printToFiji() {
        getIP().show();
    }

    /**
     * Gets the image plus for this image.
     *
     * @return The image plus for this image.
     */
    private ImagePlus getIP() {//TODO: look into multi threading this.

        ImageStack stack = dim.getImageStack();

        float[] layerImage = new float[dim.layerSize()];
        float[] layerCoherence = new float[dim.layerSize()];

        for (int t = 0; t < dim.batchSize; t++) {

            for (int z = 0; z < dim.depth; z++) {

                FloatProcessor fp = dim.getFloatProcessor();
                fp.setMinAndMax(0, Math.PI);

                image.get(z, t).getVal(handle).get(handle, layerImage);
                coherence.get(z, t).getVal(handle).get(handle, layerCoherence);

                for (int col = 0; col < dim.width; col++)
                    for (int row = 0; row < dim.height; row++) {
                        int fromInd = col * dim.height + row;
                        float pixVal = layerCoherence[fromInd] > tolerance ? layerImage[fromInd] : Float.NaN;
                        fp.setf(col, row, pixVal);
                    }
                stack.addSlice(sliceNames[z], fp);
            }
        }

        ImagePlus imp = new ImagePlus(stackName, stack);
        imp.getProcessor().setMinAndMax(0, Math.PI);

        return dim.setToHyperStack(new ImagePlus(stackName, stack));
    }

    /**
     * Saves the grayscale image stack as a hyperstack in a file.
     *
     * @param writeToFolder The folder where the image should be saved.
     */
    @Override
    public void printToFile(String writeToFolder) {//TODO: This doesn't seem to work.

        ImgPlsToFiles.saveSlices(getIP(), writeToFolder);
    }

    // Helper method to print the pixels of a FloatProcessor
    private static void printFloatProcessorPixels(FloatProcessor fp) {
        float[] pixels = (float[]) fp.getPixels(); // Get the 1D pixel array
        int w = fp.getWidth();
        int h = fp.getHeight();

        for (int row = 0; row < h; row++) {
            StringBuilder rowOutput = new StringBuilder();
            for (int col = 0; col < w; col++) {
                // Pixels are stored in row-major order within the 1D array
                // Index for (row, col) is row * width + col
                float pixelValue = pixels[row * w + col];
                rowOutput.append(String.format("%.2f ", pixelValue)); // Format to 2 decimal places for brevity
            }
            System.out.println(rowOutput.toString());
        }
    }
}
