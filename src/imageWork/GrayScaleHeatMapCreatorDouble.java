package imageWork;

import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
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
public class GrayScaleHeatMapCreatorDouble extends HeatMapCreator {

    private final PArray2dToD2d image;
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
    public GrayScaleHeatMapCreatorDouble(String[] sliceNames, String stackName, Handle handle, PArray2dToD2d image, PArray2dToF2d coherence, double tolerance, Dimensions dim) {
        super(sliceNames, stackName, dim, handle);
        
        this.image = image;

        this.coherence = coherence;
        this.tolerance = tolerance;
    }

    /**
     * Gets the image plus for this image.
     *
     * @return The image plus for this image.
     */
    @Override
    public ImageStack getStack() {//TODO: look into multi threading this.

        ImageStack stack = dim.emptyStack();

        double[] layerImage = new double[dim.layerSize()];
        float[] layerCoherence = new float[dim.layerSize()];

        for (int t = 0; t < dim.batchSize; t++) {

            for (int z = 0; z < dim.depth; z++) {

                FloatProcessor fp = dim.getFloatProcessor();
                fp.setMinAndMax(0, Math.PI);

                image.get(z, t).getVal(handle).get(handle, layerImage);
                if(coherence != null) coherence.get(z, t).getVal(handle).get(handle, layerCoherence);

                for (int col = 0; col < dim.width; col++)
                    for (int row = 0; row < dim.height; row++) {
                        int fromInd = col * dim.height + row;
                        double pixVal = coherence != null && layerCoherence[fromInd] <= tolerance ? Float.NaN : layerImage[fromInd];
                        fp.setf(col, row, (float)pixVal);
                    }
                stack.addSlice(sliceNames ==null?"":sliceNames[z], fp);
            }
        }

        return stack;
    }

    /**
     * Saves the grayscale image stack as a hyperstack in a file.
     *
     * @param writeToFolder The folder where the image should be saved.
     */
    @Override
    public void printToFile(String writeToFolder) {

        new MyImagePlus(stackName ==null?"":stackName, getStack(), dim.depth).saveSlices(writeToFolder);
    }

}
