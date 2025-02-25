package fijiPlugin;

import JCudaWrapper.array.Array;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn {

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

        ImagePlus imp = ij.WindowManager.getCurrentImage();

        if (!validImage(imp)) return;

        int defaultNeighborhoodRadius = 3;
        double defaultTolerance = 1;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");
        gd.addNumericField("Neighborhood xy radius:", defaultNeighborhoodRadius, 0);
        gd.addNumericField("Neighborhood z radius:", defaultNeighborhoodRadius, 0);
        gd.addCheckbox("use coherence", false);
        gd.showDialog();

        if (gd.wasCanceled()) return;

        NeighborhoodDim neighborhoodSize = new NeighborhoodDim((int) gd.getNextNumber(), (int) gd.getNextNumber());
        boolean useCoherence = (boolean) gd.getNextBoolean();

        if (!validParamaters(neighborhoodSize.xy) || !validParamaters(neighborhoodSize.z)) return;

        try (
                Handle handle = new Handle();
                NeighborhoodPIG np = NeighborhoodPIG.get(handle, imp, neighborhoodSize, defaultTolerance)) {

            ImageCreator ic = np.getZentihAngle(useCoherence);
            ic.printToFiji(useCoherence);
//            ic.printToFile("images/output/test2/");
            
            if(imp.getNSlices() > imp.getNFrames()) np.getZentihAngle(useCoherence).printToFiji(false);

            ij.IJ.showMessage("NeighborhoodPIG processing complete.");
        }
    }

    public static void main(String[] args) {

            String imagePath = "images/input/5Tests/";int depth = 1;
//        String imagePath = "images/input/5debugs/";
//        String imagePath = "images/input/debug/";int depth = 9;

        NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1); // Default neighborhood radius
        double tolerance = 1; // Default tolerance
        

        try (Handle handle = new Handle();
                NeighborhoodPIG np = NeighborhoodPIG.get(handle, imagePath, depth, neighborhoodSize, tolerance)) {

            np.getAzimuthalAngles(true).printToFile("images/output/test2/");
//                np.getImageOrientationXY(true).printToFiji(true);

        }
        System.out.println("NeighborhoodPIG processing complete.");
        
        System.out.println("fijiPlugin.FijiPlugin.main() - unclosed arrays: " + Array.allocatedArrays.toString());
//        System.out.println("fijiPlugin.FijiPlugin.main() not deleted " + JCudaWrapper.array.DArray.alocatedArrays);

    }

}
