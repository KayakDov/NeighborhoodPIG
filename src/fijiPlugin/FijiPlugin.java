package fijiPlugin;

import FijiInput.UserCanceled;
import FijiInput.UserInput;
import JCudaWrapper.array.Array;
import JCudaWrapper.resourceManagement.GPU;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.Toolbar;
import ij.plugin.PlugIn;
import ij.process.ImageConverter;
import imageWork.HeatMapCreator;
import imageWork.MyImagePlus;
import imageWork.MyImageStack;
import imageWork.ProcessImage;
import imageWork.VectorImg;
import java.awt.Color;
import java.io.File;
import javax.swing.SwingUtilities;
import jcuda.Sizeof;

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
     * The maximum number of frames that can be processed at once.
     *
     * @param height The height of each frame.
     * @param width The width of each frame.
     * @param depth The depth of each frame.
     * @return The maximum number of frames that can be processed at once.
     */
    public static int framesPerRun(int height, int width, int depth) {

        long freeMemory = GPU.freeMemory();

        if (freeMemory == 0) throw new RuntimeException("There is no free GPU memory.");

        long voxlesPerFrame = height * width * depth;

        return (int) ((freeMemory / voxlesPerFrame) / (Sizeof.DOUBLE * (depth > 1 ? 6 : 3) + Sizeof.FLOAT * (depth > 1 ? 3 : 2)));

    }

    @Override
    public void run(String string) {

        MyImagePlus originalImage = new MyImagePlus(ij.WindowManager.getCurrentImage());

        originalImage.setOpenAsHyperStack(true);

        if (!validImage(originalImage)) return;

        UserInput ui;

        if (string.length() == 0)try {
            ui = UserInput.fromDiolog(originalImage);
        } catch (UserCanceled ex) {
            System.out.println("fijiPlugin.FijiPlugin.run() User canceled diolog.");
            return;
        } else ui = UserInput.fromStrings(string, originalImage);
        run(ui, originalImage, false);
    }

    public static void loadImageJ() {
        // Ensure ImageJ is running. This is crucial if you're running this
        // as a standalone Java application. If it's a plugin running within Fiji,
        // ImageJ will already be initialized.
        ImageJ ij = IJ.getInstance();
        if (ij == null) {
            ij = new ImageJ(ImageJ.NO_SHOW); // Start ImageJ without showing the main window initially
        }

        // Schedule the UI update to run on the Event Dispatch Thread
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                // Get the singleton instance of the Toolbar
                Toolbar toolbar = Toolbar.getInstance();

                if (toolbar != null) {
                    toolbar.setVisible(true);
                }
            }
        });
    }

    public static void main(String[] args) {//TODO: test somethign with depth!

        loadImageJ();

//        String imagePath = "images/input/cyl/"; int depth = 250; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(8, 8, 1);
        String imagePath = "images/input/5Tests/"; int depth = 1; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(15, 1, 1);
//        String imagePath = "images/input/debug/";int depth = 1;NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1, 1);
//        String imagePath = "images/input/3dVictorData"; int depth = 1/*20*/; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(15, 1, 1);
//        String imagePath = "images/input/upDown/";int depth = 1;NeighborhoodDim neighborhoodSize = new NeighborhoodDim(1, 1);
//        String imagePath = "images/input/3dVictorDataRepeated";int depth = 20; NeighborhoodDim neighborhoodSize = new NeighborhoodDim(15, 1, 1);

        UserInput ui = UserInput.defaultVals(neighborhoodSize);

        ImagePlus imp = ProcessImage.imagePlus(imagePath, depth);

        imp = new Dimensions(imp.getStack(), depth).setToHyperStack(imp);

        imp.setDimensions(1, depth, new File(imagePath).list().length / depth);

        run(ui, imp, true);

    }

    /**
     * Allows for code to be run both from the main and from run.
     *
     * @param ui The user input.
     * @param userImg The image to be run.
     * @param toFile True if the results should be saved to a file, false if
     * they're to be displayed in fiji.
     */
    public static void run(UserInput ui, ImagePlus userImg, boolean toFile) {
        if (!ui.validParamaters())
            throw new RuntimeException("fijiPlugin.FijiPlugin.run() Invalid Parameters!");

        try {

            MyImagePlus img = new MyImagePlus(userImg).crop(
                    ui.downSample(userImg.getHeight()),
                    ui.downSample(userImg.getWidth())
            );

            Dimensions downSampled = img.dim().downSampleXY(null, ui.downSampleFactorXY);

            MyImageStack vf = VectorImg.space(downSampled, ui.vfSpacing, ui.vfMag, ui.overlay ? img.dim() : null).emptyStack(),
                    coh = downSampled.emptyStack(),
                    az = downSampled.emptyStack(),
                    zen = downSampled.emptyStack();

            int vecImgDepth = 0;

            int framesPerIteration = framesPerRun(img.dim().height, img.dim().width, img.dim().depth);
            if (framesPerIteration > 1) framesPerIteration = framesPerIteration / 2;

            System.out.println("fijiPlugin.FijiPlugin.run() frames per iteration: " + framesPerIteration);

            long startTime = System.currentTimeMillis();

            for (int i = 0; i < img.getNFrames(); i += framesPerIteration)
            try (Handle handle = new Handle(); NeighborhoodPIG np = new NeighborhoodPIG(handle, img.subset(i, framesPerIteration), ui)) {

                if (ui.heatMap) {
                    appendHM(az, np.getAzimuthalAngles(0.01), 0, (float) Math.PI);
                    if (img.dim().hasDepth()) appendHM(zen, np.getZenithAngles(false, 0.01), 0, (float) Math.PI);
                }

                if (ui.vectorField)
                    vecImgDepth = appendVF(
                            ui,
                            np.getVectorImg(ui.vfSpacing, ui.vfMag, false, ui.overlay ? img.dim() : null), 
                            vf
                    );

                if (ui.useCoherence) appendHM(coh, np.getCoherence(ui.tolerance), 0, 1);

            }

            long endTime = System.currentTimeMillis();

            System.out.println("Execution time: " + (endTime - startTime) + " milliseconds");

            results(vf, coh, az, zen, toFile, ui, img.dim(), vecImgDepth, img);
        } catch (Exception ex) {
            System.out.println("fijiPlugin.FijiPlugin.run() " + ui.toString());
            throw ex;
        }
        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak.");
    }

    /**
     * Appends to the stack from the vector field.
     *
     * @param ui The user input.
     * @param vecImg The vector image to be added to the stack.
     * @param vf The stack to be appended to.
     * @return The depth of the vector image.
     */
    public static int appendVF(UserInput ui, VectorImg vecImg, MyImageStack vf) {
        vf.concat(vecImg.imgStack());
        return vecImg.getOutputDimensions().depth;
    }

    /**
     * Appends the heatmap to the stack.
     *
     * @param addTo The stack to have the heatmap's stack added onto it.
     * @param add The heatmap whose stack is to be appended.
     */
    public static void appendHM(MyImageStack addTo, HeatMapCreator add, float min, float max) {
        addTo.concat(add.getStack(min, max));
    }

    /**
     * Presents the images.
     *
     * @param vf The vector field.
     * @param coh The coherence.
     * @param az The azimuthal angles.
     * @param zen The zenith angles.
     * @param toFile True if images should be saved to a file, false if they
     * should be shown in Fiji.
     * @param ui The user input.
     * @param dims The dimensions.
     * @param vecImgDepth The depth of the vector image.
     * @param myImg The image worked on.
     */
    private static void results(MyImageStack vf, MyImageStack coh, MyImageStack az, MyImageStack zen, boolean toFile, UserInput ui, Dimensions dims, int vecImgDepth, MyImagePlus myImg) {

        if (ui.heatMap) {
            present(az.imp("Azimuthal Angles", dims.depth), toFile, "images/output/test3/Azimuthal");
            if (dims.hasDepth())
                present(zen.imp("Zenith Angles", dims.depth), toFile, "images/output/test3/Zenith");
        }
        if (ui.vectorField) {
            MyImagePlus impVF;
            if (!dims.hasDepth() && ui.overlay)
                impVF = new MyImagePlus("Overlaid Nematic Vectors", myImg.getImageStack(), dims.depth)
                        .overlay(vf, Color.GREEN);
            else impVF = vf.imp("Nematic Vectors", vecImgDepth);
            present(impVF, toFile, "images/output/test3/vectors");
        }

        if (ui.useCoherence) present(coh.imp("Coherence", dims.depth), toFile, "images/output/test3/Coherence");
    }

    /**
     * Presents the image, either by showing it on Fiji or saving it as a file.
     *
     * @param image The image to be presented.
     * @param saveTo The location to save the image. This will not be used if
     * show is set to true.
     */
    private static void present(MyImagePlus image, boolean toFile, String saveTo) {
        if (toFile) image.saveSlices(saveTo);
        else image.show();
    }

}
