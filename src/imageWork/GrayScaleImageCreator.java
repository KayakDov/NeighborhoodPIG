package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.io.FileSaver;
import java.io.File;

/**
 * A class for creating grayscale images from tensor data.
 *
 * This class processes 3D tensor data to generate grayscale intensity maps. It
 * supports GPU-accelerated processing and provides functionality for displaying
 * and saving images.
 *
 * @author E. Dov Neimand
 */
public class GrayScaleImageCreator extends ImageCreator {

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
     */
    public GrayScaleImageCreator(String[] sliceNames, String stackName, Handle handle, FStrideArray3d image) {
        super(sliceNames, stackName, handle, image);
        pixelIntensity = image.get(handle);
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
    private ImagePlus getIP() {
        ImageStack stack = new ImageStack(width, height);
        FloatProcessor fp = new FloatProcessor(width, height);

        for (int t = 0; t < batchSize; t++) {
            int frameInd = t * width * height * depth;
            for (int z = 0; z < depth; z++) {
                int layerInd = frameInd + z * width * height;
                
                for (int col = 0; col < width; col++) {
                    int colInd = layerInd + col * height;
                    for (int row = 0; row < height; row++)
                        fp.setf(col, row, pixelIntensity[colInd + row]);

                    stack.addSlice(
                            sliceNames[z],
                            fp
                    );
                }
            }
        }

        ImagePlus image = new ImagePlus(stackName, stack);
        image.setDimensions(1, depth, batchSize);
        return image;
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
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        String filePath = new File(outputDir, stackName + ".tif").getAbsolutePath();
        FileSaver saver = new FileSaver(image);
        saver.saveAsTiff(filePath);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
