package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.io.FileSaver;
import java.io.File;
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

    /**
     * Array storing pixel intensity values.
     */
    private final float[] pixelIntensity;

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
    public GrayScaleHeatMapCreator(String[] sliceNames, String stackName, Handle handle, FStrideArray3d image, FStrideArray3d coherence, double tolerance) {
        super(sliceNames, stackName, handle, image);
        pixelIntensity = image.get(handle);
        if (coherence != null) {
            float[] coh = coherence.get(handle);
            IntStream.range(0, coh.length).filter(i -> coh[i] <= tolerance).forEach(i -> pixelIntensity[i] = Float.NaN);
        }
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

        for (int t = 0; t < batchSize; t++) {
            int frameInd = t * width * height * depth;

            for (int z = 0; z < depth; z++) {

                FloatProcessor fp = new FloatProcessor(width, height);
                int layerInd = frameInd + z * width * height;

                for (int col = 0; col < width; col++) {
                    int colInd = layerInd + col * height;
                    for (int row = 0; row < height; row++)
                        fp.setf(col, row, pixelIntensity[colInd + row]);
                }
                stack.addSlice(
                        sliceNames[z],
                        fp//.convertToByte(true)
                );

            }
        }

        return setToHyperStack(new ImagePlus(stackName, stack));
    }

    /**
     * Saves the grayscale image stack as a hyperstack in a file.
     *
     * @param writeToFolder The folder where the image should be saved.
     */
    @Override
    public void printToFile(String writeToFolder) {

        ImagePlus image = getIP();

        File outputDir = new File(writeToFolder);
        if (!outputDir.exists()) outputDir.mkdirs();

        String filePath = new File(outputDir, stackName + ".tif").getAbsolutePath();
        FileSaver saver = new FileSaver(image);
        saver.saveAsTiff(filePath);
    }

}
