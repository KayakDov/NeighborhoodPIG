package fijiPlugin;

import FijiInput.UserCanceled;
import FijiInput.UserInput;
import JCudaWrapper.array.Array;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.plugin.PlugIn;
import ij.process.ImageConverter;
import imageWork.ProcessImage;

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

    //TODO (1) remove NaN vaslues.  (2) get same results as OrientationJ.
    
    @Override
    public void run(String string) {
        
        ImagePlus imp = ij.WindowManager.getCurrentImage();

        imp.setOpenAsHyperStack(true);

        if (!validImage(imp)) return;
        
        UserInput ui;
        try { ui = UserInput.fromDiolog(imp);
        } catch (UserCanceled ex) {
            System.out.println("fijiPlugin.FijiPlugin.run() User canceled diolog.");
            return;
        }

        if (!ui.validParamaters()){
            System.out.println("fijiPlugin.FijiPlugin.run() Invalid Parameters!");
            return;
        }

        
        
        try (Handle handle = new Handle(); NeighborhoodPIG np = new NeighborhoodPIG(handle, imp, ui)) {           
            
            if (ui.heatMap) {
                np.getAzimuthalAngles(false, 0.01).printToFiji();

                if (imp.getNSlices() > 1) np.getZenithAngles(false, 0.01).printToFiji();
            }

            if (ui.vectorField)
                np.getVectorImg(ui.vfSpacing, ui.vfMag, false).show();

            if (ui.useCoherence) np.getCoherence().printToFiji();

        }
        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak.");
    }

    /**
     * The main to be run if there are no command line arguments.
     */
    public static void defaultRun() {

        
        
//        String imagePath = "images/input/cyl/"; int depth = 50; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(4, 1, 1);
//        String imagePath = "images/input/5Tests/"; int depth = 1; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(15, 1, 1);
        String imagePath = "images/input/debug/";int depth = 5;NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1, 1);
//            String imagePath = "images/input/3dVictorData";int depth = 20; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(30, 1, 1);
//        String imagePath = "images/input/upDown/";int depth = 1;NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1);

        UserInput ui = UserInput.defaultVals(neighborhoodSize);

        ImagePlus imp = ProcessImage.imagePlus(imagePath, depth);
        
        try (Handle handle = new Handle(); NeighborhoodPIG np = new NeighborhoodPIG(handle, imp, ui)) {
                        
            np.getAzimuthalAngles(false, .01).printToFile("images/output/test3/Azimuthal");

            if (depth > 1) np.getZenithAngles(false, .01).printToFile("images/output/test3/Zenith");

//            np.getVectorImg(8, 6, false);

        }        

        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak. " + Array.allocatedArrays.size() + " arrays remain allocated.  They are " + Array.allocatedArrays.toString());
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            defaultRun();
            return;
        }


    }

}
