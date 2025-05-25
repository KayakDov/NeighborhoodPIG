package imageWork;

import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import java.util.Arrays;
import java.util.stream.IntStream;

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
    private final PArray2dToD2d coherence;
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
     */
    public GrayScaleHeatMapCreator(String[] sliceNames, String stackName, Handle handle, PArray2dToD2d image, PArray2dToD2d coherence, double tolerance) {
        super(sliceNames, stackName, handle, image);

        this.image = new PArray2dToF2d(image.entriesPerLine(), image.linesPerLayer(), image.targetDim().entriesPerLine, image.targetDim().numLines);
        this.image.initTargets(handle);

        this.coherence = coherence;
        this.tolerance = tolerance;

        Kernel.run("mapToFloat", handle,
                size(),
                new PArray2dTo2d[]{this.image, image},
                this
        );

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

        ImageStack stack = new ImageStack(width, height);

        float[] layerImage = new float[layerSize()];
        double[] layerCoherence = new double[layerSize()];

        for (int t = 0; t < batchSize; t++) {

            for (int z = 0; z < depth; z++) {

                FloatProcessor fp = new FloatProcessor(width, height);
//                fp.setMinAndMax(0, Math.PI);//TODO:not sure if I need this line.

                image.get(z, t).getVal(handle).get(handle, layerImage);
                coherence.get(z, t).getVal(handle).get(handle, layerCoherence);

                for (int col = 0; col < width; col++)
                    for (int row = 0; row < height; row++) {
                        int fromInd = col * height + row;

                        fp.setf(col, row, layerImage[fromInd] * (layerCoherence[fromInd] <= tolerance ? 0f : 1f));
                    }
                stack.addSlice(sliceNames[z], fp);

                // --- ADD THIS SECTION TO PRINT PIXEL VALUES ---
                System.out.println("--- Pixels for Frame " + (t + 1) + ", Z-Slice " + (z + 1) + " ---");
                printFloatProcessorPixels(fp);
                System.out.println("--- End Pixels ---");
                // --- END ADDITION ---

            }
        }

        ImagePlus imp = new ImagePlus(stackName, stack);
        imp.getProcessor().setMinAndMax(0, Math.PI);

        return setToHyperStack(new ImagePlus(stackName, stack));
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

    @Override
    public void close() {
        image.close();
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
