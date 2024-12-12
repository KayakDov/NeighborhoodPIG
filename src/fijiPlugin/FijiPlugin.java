package fijiPlugin;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import org.apache.commons.math3.complex.Complex;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn {

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
     * @param imp The image.
     * @return true if the image is selected and gray scale, false otherwise.
     */
    private static boolean validImage(ImagePlus imp){
        
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
     * @param nRad The neighborhood radius.
     * @param tol The tolerance.
     * @return true if the parameters are valid, false otherwise.
     */
    private static boolean validParamaters(int nRad, double tol){

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

        ImagePlus imp = ij.WindowManager.getCurrentImage();

        if(!validImage(imp)) return;

        int defaultNeighborhoodRadius = 3;
        double defaultTolerance = 1e-10;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");
        gd.addNumericField("Neighborhood raqdius:", defaultNeighborhoodRadius, 0);
        gd.addNumericField("Tolerance:", defaultTolerance, 2);
        gd.showDialog();
        
        if (gd.wasCanceled()) return;
        

        int neighborhoodSize = (int) gd.getNextNumber();
        double tolerance = gd.getNextNumber();

        if(!validParamaters(neighborhoodSize, tolerance)) return;

        NeighborhoodPIG np = new NeighborhoodPIG(imp, neighborhoodSize, tolerance);

        ij.IJ.showMessage("NeighborhoodPIG processing complete.");
    }

}
