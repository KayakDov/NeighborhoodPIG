package fijiPlugin;

import JCudaWrapper.array.Array;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.ImageConverter;
import ij3d.Image3DUniverse;

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

        // Convert the image to grayscale if it's not already
        if (imp.getType() != ImagePlus.GRAY32
                && imp.getType() != ImagePlus.GRAY16
                && imp.getType() != ImagePlus.GRAY8) {
            ij.IJ.showMessage("Image is being converted to grayscale.");
            ImageConverter converter = new ImageConverter(imp);
            converter.convertToGray32(); // Convert to 32-bit grayscale for consistency
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

        if (nRad < 0) {
            ij.IJ.showMessage("Neighborhood size must be a positive number.");
            return false;
        }

        return true;
    }

    @Override
    public void run(String string) {

        ImagePlus imp = ij.WindowManager.getCurrentImage();

        imp.setOpenAsHyperStack(true);

        if (!validImage(imp))
            return;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");
        gd.addNumericField("Distance between layers as a multiple of the distance between pixels:", 1, 2);
        gd.addNumericField("Neighborhood xy radius:", 30, 2);
        if (imp.getNSlices() > 1)
            gd.addNumericField("Neighborhood z radius:", 1, 2);
        gd.addCheckbox("generate coherence", false);
        gd.showDialog();

        if (gd.wasCanceled())
            return;

        float layerDist = (float) gd.getNextNumber();

        NeighborhoodDim neighborhoodSize = new NeighborhoodDim((int) gd.getNextNumber(), imp.getNSlices() > 1 ? (int) gd.getNextNumber() : 0, layerDist);
        boolean useCoherence = (boolean) gd.getNextBoolean();

        if (!validParamaters(neighborhoodSize.xyR) || !validParamaters(neighborhoodSize.zR))
            return;

        try (
                Handle handle = new Handle(); NeighborhoodPIG np = NeighborhoodPIG.get(handle, imp, neighborhoodSize, 1)) {

            np.getAzimuthalAngles(false, false).printToFiji();

            if (imp.getNSlices() > 1)
                np.getZenithAngles(false, false).printToFiji();

            np.getVectorImg(20, 8).show();
            if (useCoherence)
                np.getAzimuthalAngles(false, true).printToFiji();

            ij.IJ.showMessage("NeighborhoodPIG processing complete.");
        }
        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak.");
    }

    /**
     * The main to be run if there are no command line arguments.
     */
    public static void defaultRun(){
        
//        String imagePath = "images/input/5Tests/"; int depth = 1; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(4, 1, 1);
        String imagePath = "images/input/5debugs/";
        int depth = 9;
        NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1, 1);
//        String imagePath = "images/input/debug/";int depth = 1;NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1, 1);
//            String imagePath = "images/input/3dVictorData";int depth = 20; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1);
//        String imagePath = "images/input/upDown/";int depth = 1;NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1);

        float tolerance = 10f; // Default tolerance

        try (Handle handle = new Handle(); NeighborhoodPIG np = NeighborhoodPIG.get(handle, imagePath, depth, neighborhoodSize, tolerance)) {

            np.getAzimuthalAngles(true, true).printToFile("images/output/test3/Azimuthal");

            if (depth > 1)
                np.getZenithAngles(true, true).printToFile("images/output/test3/Zenith");

            np.getVectorImg(5, 5).show();

        }
        System.out.println("NeighborhoodPIG processing complete.");

        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak.");

    }
    
    
    public static void main(String[] args) {
        if(args.length == 0){
            defaultRun();
            return;
        }
        
    }

}
