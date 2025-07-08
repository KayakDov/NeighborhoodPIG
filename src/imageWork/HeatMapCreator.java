package imageWork;

import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import main.Test;

/**
 * A class for creating grayscale images from tensor data.
 *
 * This class processes 3D tensor data to generate grayscale intensity maps. It
 * supports GPU-accelerated processing and provides functionality for displaying
 * and saving images.
 *
 * @author E. Dov Neimand
 */
public class HeatMapCreator{

    private final PArray2dToF2d image;
    private final PArray2dToF2d coherence;
    private final double tolerance;
    /**
     * Array storing color data for each tensor element.
     */
    protected final String[] sliceNames;
    protected final String stackName;
    protected final Dimensions dim;
    protected final Handle handle;

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
     * @param tolerance Defines what coherence close enough to 0 such that it
     * will appear as NaN rather than an orientation.
     */
    public HeatMapCreator(String[] sliceNames, String stackName, Handle handle, PArray2dToF2d image, PArray2dToF2d coherence, double tolerance) {

        this.image = image;
        this.coherence = coherence;
        this.tolerance = tolerance;
        this.sliceNames = sliceNames;
        this.stackName = stackName;
        dim = new Dimensions(image);
        this.handle = handle;

    }

    /**
     * Gets the image plus for this image.
     *
     * @param min The minimum value in the range.
     * @param max The maximum value in the range.
     * @param es An executor service.
     * @return The image plus for this image.
     */
    public ImageStack getStack(float min, float max, ExecutorService es) {//TODO: look into multi threading this.

//        System.out.println("imageWork.HeatMapCreator.getStack() 500 500 " + image.get(0,0).getVal(handle).get(500, 500).getf(handle));
        MyImageStack stack = dim.emptyStack();

        FloatProcessor[] slices = new FloatProcessor[dim.totLayers()];
        Arrays.parallelSetAll(slices, i -> dim.getFloatProcessor(min, max));


        for (int t = 0; t < dim.batchSize; t++) {

            for (int z = 0; z < dim.depth; z++) {

                float[] layerImage = new float[dim.layerSize()];
                float[] layerCoherence = new float[dim.layerSize()];
                FloatProcessor fp = slices[t * dim.depth + z];

                image.get(z, t).getVal(handle).get(handle, layerImage);

                if (coherence != null) coherence.get(z, t).getVal(handle).get(handle, layerCoherence);

//                processLayer(fp, layerCoherence, layerImage).run();
                es.submit(processLayer(fp, layerCoherence, layerImage));
            }
        }
        

        for (int i = 0; i < slices.length; i++)
            stack.addSlice(sliceNames == null ? "" : sliceNames[i], slices[i]);
        
        return stack;
    }

    /**
     * transfers a layer from a cpu array to the float processor.
     *
     * @param fp The data destination.
     * @param layerCoherence The coherence of the data. Pixels with near-0
     * coherence will be set to NaN.
     * @param layerImage
     * @return
     */
    private Runnable processLayer(FloatProcessor fp, float[] layerCoherence, float[] layerImage) {
        return () -> {
            for (int x = 0; x < dim.width; x++)
                for (int y = 0; y < dim.height; y++) {
                    int fromInd = x * dim.height + y;

                    float pixVal = coherence == null || layerCoherence[fromInd] > tolerance ? layerImage[fromInd] : Float.NaN;
                    fp.setf(x, y, pixVal);
                }

        };
    }

    // Helper method to print the pixels of a FloatProcessor
    private static void printFloatProcessorPixels(FloatProcessor fp) {
        float[] pixels = (float[]) fp.getPixels(); // Get the 1D pixel array
        int w = fp.getWidth();
        int h = fp.getHeight();

        for (int row = 0; row < h; row++) {
            StringBuilder rowOutput = new StringBuilder();
            for (int col = 0; col < w; col++) {

                float pixelValue = pixels[row * w + col];
                rowOutput.append(String.format("%.2f ", pixelValue)); // Format to 2 decimal places for brevity
            }
            System.out.println(rowOutput.toString());
        }
    }

   
    
//    @Override
//    public void close() throws Exception {
//        try {
//            boolean terminated = es.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); // Wait indefinitely
//            if (!terminated) {
//                System.err.println("HeatMapCreator: Executor did not terminate in time. Some image processing tasks might be incomplete.");
//                // Handle this error case: perhaps log it, throw an exception, or attempt to force shutdown.
//                // executor.shutdownNow(); // Consider forceful shutdown if tasks hang
//            }
//        } catch (InterruptedException e) {
//            // If the current thread (the one calling getStack) is interrupted while waiting
//            Thread.currentThread().interrupt(); // Restore the interrupted status
//            System.err.println("HeatMapCreator: Waiting for image processing tasks was interrupted.");
//            throw new RuntimeException("Image processing interrupted", e); // Re-throw as unchecked exception or handle
//        }
//    }

    
}
