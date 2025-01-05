package fijiPlugin;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn {

    private ImagePlus imp;


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
    private static boolean validParamaters(int nRad) {

        if (nRad <= 0) {
            ij.IJ.showMessage("Neighborhood size must be a positive number.");
            return false;
        }

        return true;
    }

    @Override
    public void run(String string) {

        imp = ij.WindowManager.getCurrentImage();

        if (!validImage(imp)) return;

        int defaultNeighborhoodRadius = 3;
        double defaultTolerance = 1e-11;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");
        gd.addNumericField("Neighborhood radius:", defaultNeighborhoodRadius, 0);        
        gd.showDialog();

        if (gd.wasCanceled()) return;

        int neighborhoodSize = (int) gd.getNextNumber();        

        if (!validParamaters(neighborhoodSize)) return;

        NeighborhoodPIG np = new NeighborhoodPIG(imp, neighborhoodSize, defaultTolerance);

        np.getImageOrientationXY().printToFiji();
        np.getImageOrientationYZ().printToFiji();

        ij.IJ.showMessage("NeighborhoodPIG processing complete.");
    }
    

    public static void main(String[] args) {

//        String imagePath = "images/input/test.jpeg";
        String imagePath = "images/input/debug.jpeg";
        ImagePlus imp = loadImageAsStack(imagePath);

        int neighborhoodSize = 10; // Default neighborhood radius
        double tolerance = 1e-10; // Default tolerance

        NeighborhoodPIG np = new NeighborhoodPIG(imp, neighborhoodSize, tolerance);


        np.getImageOrientationXY().printToFile("images/output/test2/");

        np.close();
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
