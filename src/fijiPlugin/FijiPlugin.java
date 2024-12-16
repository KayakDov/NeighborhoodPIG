package fijiPlugin;

import JCudaWrapper.array.IArray;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.plugin.ImagesToStack;
import ij.plugin.PlugIn;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;
import java.io.File;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import org.apache.commons.math3.complex.Complex;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn {

    private ImagePlus imp;

    public static void sampleFijiCode() {//TODO: delte me

        IJ.showMessage("Hello, World! ", " Welcome Ahhhh! Fiji plugin development! ");

        ImagePlus imp = ij.WindowManager.getCurrentImage();
        if (imp == null) {
            ij.IJ.showMessage("No image open.");
            return;
        }

        if (imp.getType() != ImagePlus.COLOR_RGB) {
            ij.IJ.showMessage("No image open.");
            IJ.run(imp, "RGB Color", ""); // Convert to RGB
        }

        int width = imp.getWidth();
        int height = imp.getHeight();
        int depth = imp.getStackSize();

        ij.IJ.showMessage("There is an open image.");
        for (int z = 1; z <= depth; z++) {
            ImageProcessor ip = imp.getStack().getProcessor(z);

            // Iterate through each pixel in the slice
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Check if the pixel is "every other" pixel based on a pattern (x + y)
                    if ((x + y) % 2 == 0) {
                        // Change pixel color to blue (in RGB format)
                        ip.putPixel(x, y, (255 << 16) | (255 << 8)); // Blue with full intensity, red and green are zero
                    }
                }
            }
        }

        imp.updateAndDraw();
        imp.show();
    }

    /**
     * Checks that the image is selected and gray scale.
     *
     * @param imp The image.
     * @return true if the image is selected and gray scale, false otherwise.
     */
    private static boolean validImage(ImagePlus imp) {

        // Check if an image is open
        if (imp == null) {
            ij.IJ.showMessage("No image open.");
            return false;
        }

        // Ensure the image is grayscale
        if (imp.getType() != ImagePlus.GRAY32
                && imp.getType() != ImagePlus.GRAY16
                && imp.getType() != ImagePlus.GRAY8) {
            ij.IJ.showMessage("Image needs to be grayscale.");
            return false;
        }

        return true;
    }

    /**
     * Checks if the parameters are valid.
     *
     * @param nRad The neighborhood radius.
     * @param tol The tolerance.
     * @return true if the parameters are valid, false otherwise.
     */
    private static boolean validParamaters(int nRad, double tol) {

        if (nRad <= 0) {
            ij.IJ.showMessage("Neighborhood size must be a positive number.");
            return false;
        }

        if (tol < 0) {
            ij.IJ.showMessage("Tolerance must be non-negative.");
            return false;
        }

        return true;
    }

    @Override
    public void run(String string) {

        imp = ij.WindowManager.getCurrentImage();

        if (!validImage(imp)) return;

        int defaultNeighborhoodRadius = 3;
        double defaultTolerance = 1e-10;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");
        gd.addNumericField("Neighborhood raqdius:", defaultNeighborhoodRadius, 0);
        gd.addNumericField("Tolerance:", defaultTolerance, 2);
        gd.showDialog();

        if (gd.wasCanceled()) return;

        int neighborhoodSize = (int) gd.getNextNumber();
        double tolerance = gd.getNextNumber();

        if (!validParamaters(neighborhoodSize, tolerance)) return;

        NeighborhoodPIG np = new NeighborhoodPIG(imp, neighborhoodSize, tolerance);

        np.fijiDisplayOrientationHeatmap();

        ij.IJ.showMessage("NeighborhoodPIG processing complete.");
    }

    public static void main(String[] args) {

        String imagePath = "images/input/test.jpeg";
        ImagePlus imp = loadImageAsStack(imagePath);

        int neighborhoodSize = 3; // Default neighborhood radius
        double tolerance = 1e-10; // Default tolerance

        NeighborhoodPIG np = new NeighborhoodPIG(imp, neighborhoodSize, tolerance);

        np.fijiDisplayOrientationHeatmap();

        System.out.println("NeighborhoodPIG processing complete.");
    }

    /**
     * Loads an image from the given path, converts it to grayscale, and ensures
     * it is of type 32-bit float.
     *
     * @param imagePath The path to the input image.
     * @return An ImagePlus object containing a stack of the processed image.
     */
    private static ImagePlus loadImageAsStack(String imagePath) {

        ImagePlus imp = new ImagePlus(imagePath);

        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32) 
            imp = new ImagePlus("Grayscale", imp.getProcessor());
        

        // Ensure it is 32-bit float
        ImageProcessor floatProcessor = imp.getProcessor().convertToFloat();

        // Create a stack and add the processed slice
        ImageStack stack = new ImageStack(floatProcessor.getWidth(), floatProcessor.getHeight());
        stack.addSlice(floatProcessor);

        return new ImagePlus("Generated Stack (32-bit Float)", stack);
    }

}
