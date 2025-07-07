package fijiPlugin;

import FijiInput.UserCanceled;
import FijiInput.UserDialog;
import FijiInput.UserInput;
import JCudaWrapper.array.Array;
import JCudaWrapper.resourceManagement.GPU;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Toolbar;
import ij.plugin.PlugIn;
import ij.process.ImageConverter;
import imageWork.DatSaver;
import imageWork.HeatMapCreator;
import imageWork.MyImagePlus;
import imageWork.MyImageStack;
import imageWork.ProcessImage;
import imageWork.VecManager2d;
import imageWork.VectorImg;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import javax.swing.SwingUtilities;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn {

    /**
     * Defines the available save formats for plugin outputs.
     * <ul>
     * <li>{@code tiff}: Save output images as TIFF files.</li>
     * <li>{@code png}: Save output images as PNG files.</li>
     * <li>{@code fiji}: Display outputs directly within Fiji/ImageJ.</li>
     * <li>{@code txt}: Save raw vector data (x, y, z, nx, ny, nz) to a text
     * file.</li>
     * </ul>
     */
    public enum Save {
        tiff, png, fiji, txt // Added txt for saving raw vector data
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

        long voxlesPerFrame = (long) height * width * depth;

        return (int) ((freeMemory / voxlesPerFrame) / (Sizeof.DOUBLE * (depth > 1 ? 6 : 3) + Sizeof.FLOAT * (depth > 1 ? 3 : 2)));

    }

    /**
     * Executes the Neighborhood PIG plugin. This method is the entry point when
     * the plugin is invoked from the Fiji/ImageJ user interface or via a macro.
     * It can operate in two modes: by displaying a dialog for user input, or by
     * parsing parameters directly from a provided string.
     *
     * <p>
     * If the {@code string} argument is empty, a graphical user interface
     * (dialog box) will be presented to the user to gather all necessary input
     * parameters.
     * </p>
     * <p>
     * If the {@code string} argument is not empty, it is expected to contain
     * space-separated parameters that will be parsed directly. The structure
     * and order of these parameters are critical and depend on the active
     * image's dimensionality (2D or 3D) and the chosen output options (e.g.,
     * generating a vector field).
     * </p>
     * <p>
     * For the precise format and conditional inclusion of parameters when
     * providing them as a string, please refer to the Javadoc of the
     * {@link UserInput#fromStrings(String[], int) UserInput.fromStrings}
     * method.
     * </p>
     *
     * @param string A string containing space-separated user input parameters,
     * or an empty string to prompt the user with a dialog.
     * @throws NullPointerException if no image is currently open in ImageJ when
     * the plugin is launched.
     * @throws NumberFormatException if any string parameter cannot be parsed
     * into its corresponding numeric or boolean type when using string-based
     * input.
     * @throws ArrayIndexOutOfBoundsException if the {@code string} array does
     * not contain enough elements for the required parameters based on the
     * image properties and boolean flags.
     * @throws RuntimeException for other issues during image processing or if
     * internal parameters are inconsistent (e.g., spacing/downsample mismatch
     * if vector field overlay is true).
     */
    @Override
    public void run(String string) {

        MyImagePlus originalImage = null;

        try {
            originalImage = new MyImagePlus(ij.WindowManager.getCurrentImage());
        } catch (NullPointerException npe) {
            IJ.error("Missing Image", "No image found. Please open one.");
            return;
        }

        originalImage.setOpenAsHyperStack(true);

        if (!validImage(originalImage)) return;

        UserInput ui;

        if (string.length() == 0) {
            try {
                ui = new UserDialog(originalImage).getUserInput();
            } catch (UserCanceled ex) {
                System.out.println("fijiPlugin.FijiPlugin.run() User canceled dialog.");
                return;
            }
        } else {
            ui = UserInput.fromStrings(string.split(" "), originalImage.getNSlices());
        }
        run(ui, originalImage, Save.fiji);
    }

    /**
     * Loads image J. This should be called if this is not being run from Fiji.
     */
    public static void loadImageJ() {
        ImageJ ij = IJ.getInstance();
        if (ij == null) {
            ij = new ImageJ(ImageJ.NO_SHOW); // Start ImageJ without showing the main window initially
        }

        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                Toolbar toolbar = Toolbar.getInstance();

                if (toolbar != null) {
                    toolbar.setVisible(true);
                }
            }
        });
    }

    /**
     * Main entry point for running the Neighborhood PIG plugin as a standalone
     * application from the command line. This method processes images based on
     * provided arguments, allowing for headless execution without the
     * Fiji/ImageJ graphical user interface.
     *
     * <p>
     * The expected command-line arguments are structured as follows:</p>
     * <ol>
     * <li>`args[0]` (String): The path to the image file or a directory
     * containing image files to be processed as a stack. If a directory, all
     * image files within it are assumed to form a single stack (sorted
     * alphabetically).</li>
     * <li>`args[1]` (int): The nominal depth (number of Z-slices) of the image
     * or image stack. For 2D images, this should be `1`. This value determines
     * which subsequent parameters related to Z-dimensions are expected.</li>
     * <li>`args[2]` (String): "tiff" to save as a .tiff, "png" to save as a
     * ".png", or "txt" to indicate raw vector data should be saved to a text
     * file.</li>
     * <li>`args[3]` onwards (String...): A variable-length series of
     * space-separated strings representing the `UserInput` parameters. The
     * number and order of these parameters depend on the `depth` (from
     * `args[1]`) and other boolean flags (like `generate_vector_field`). Refer
     * to the
     * {@link FijiInput.UserInput#fromStrings(String[], int) UserInput.fromStrings}
     * method's Javadoc for the precise conditional order and meaning of these
     * parameters.
     * </li>
     * </ol>
     *
     * <p>
     * Example command-line usage (assuming the JAR is executable and named
     * `NeighborhoodPIG.jar`):
     * </p>
     * <pre>
     * // For a 2D image (e.g., "my_image.tif") generating a heatmap and coherence:
     * // Args: image_path, depth=1, save_format="png", xy_rad=10, z_mult=1, heatmap=true, vector_field=false, coherence=true, save_vectors=false, vf_spacing_xy=0, vf_spacing_z=0, vf_mag=0, overlay=false, downsample_xy=1, downsample_z=1
     * java -jar NeighborhoodPIG.jar "path/to/my_image.tif" 1 "png" 10 1 true false true false 0 0 0 false 1 1
     *
     * // For a 3D image stack (e.g., from "my_image_directory/") indicating raw vector data should be saved:
     * // Args: image_path, depth=5, save_format="txt", xy_rad=15, z_rad=3, z_mult=2, heatmap=false, vector_field=true, coherence=false, save_vectors=true, vf_spacing_xy=10, vf_spacing_z=8, vf_mag=2, overlay=false, downsample_xy=2, downsample_z=1
     * java -jar NeighborhoodPIG.jar "path/to/my_image_directory" 5 "txt" 15 3 2 false true false true 10 8 2 false 2 1
     *
     * // For a 2D image "another_2d_image.tif" with vector field overlaid (downsample_xy and vf_spacing_xy match):
     * // Args: image_path, depth=1, save_format="tiff", xy_rad=10, z_mult=1, heatmap=false, vector_field=true, coherence=false, save_vectors=false, vf_spacing_xy=15, vf_mag=10, overlay=true, downsample_xy=15, downsample_z=1
     * java -jar NeighborhoodPIG.jar "path/to/another_2d_image.tif" 1 "tiff" 10 1 false true false false 15 10 true 15 1
     * </pre>
     *
     * @param args Command-line arguments as described above.
     * @throws NumberFormatException if any argument cannot be parsed into its
     * corresponding numeric or boolean type.
     * @throws ArrayIndexOutOfBoundsException if the number of command-line
     * arguments is fewer than expected based on the `depth` and other boolean
     * flags.
     * @throws NullPointerException if `args[0]` points to a non-existent
     * directory or if `File.list()` returns `null` for other reasons during
     * stack dimensioning.
     * @throws RuntimeException for other issues during image processing or if
     * internal parameters are inconsistent (e.g., spacing/downsample mismatch
     * if `overlay` is true).
     */
    public static void main(String[] args) {

        if (args.length == 0) args = defaultArgs();

//        loadImageJ();
        int depth = Integer.parseInt(args[1]);

        UserInput ui = UserInput.fromStrings(Arrays.copyOfRange(args, 3, args.length), depth);

        ImagePlus imp = ProcessImage.imagePlus(args[0], depth);

        imp = new Dimensions(imp.getStack(), depth).setToHyperStack(imp);

        imp.setDimensions(1, depth, new File(args[0]).list().length / depth);

        run(ui, imp, Save.valueOf(args[2]));

    }

    private static String[] defaultArgs() {
        String imagePath = "images/input/cyl/";
        int depth = 50;
        NeighborhoodDim neighborhoodSize = new NeighborhoodDim(3, 3, 1);

        double zDist = 1;
        boolean hasHeatMap = false;
        boolean hasVF = false;
        boolean hasCoherence = false;
        boolean saveVectors = true;
        int vfSpacingXY = 6;
        int vfSpacingZ = 6;
        int mag = 4;
        boolean overlay = false;
        int downSampleXY = 1;
        int downSampleZ = 1;

        return new String[]{
            imagePath,
            "" + depth,
            "png", // Default output format for image-based results
            "" + neighborhoodSize.xyR,
            "" + neighborhoodSize.zR,
            "" + zDist,
            "" + hasHeatMap,
            "" + hasVF,
            "" + hasCoherence,
            "" + saveVectors, // This flag would be parsed by UserInput.fromStrings
//            "" + vfSpacingXY,
//            "" + vfSpacingZ,
//            "" + mag,
//            "" + overlay,
            "" + downSampleXY,
            "" + downSampleZ
        };

    }

    /**
     * Allows for code to be run both from the main and from run.
     *
     * @param ui The user input.
     * @param userImg The image to be run.
     * @param save Where to save the file to.
     */
    public static void run(UserInput ui, ImagePlus userImg, Save save) {
        if (!ui.validParamaters())
            throw new RuntimeException("fijiPlugin.FijiPlugin.run() Invalid Parameters!");

        try {

            MyImagePlus img = new MyImagePlus(userImg).crop(
                    ui.downSampleXY(userImg.getHeight()),
                    ui.downSampleXY(userImg.getWidth()),
                    ui.downSampleZ(userImg.getNSlices())
            );

            Dimensions downSampled = img.dim().downSample(null, ui.downSampleFactorXY, ui.downSampleFactorZ);

            MyImageStack vf = VectorImg.space(downSampled, ui.vfSpacingXY, ui.vfSpacingZ, ui.vfMag, ui.overlay ? img.dim() : null).emptyStack(),
                    coh = downSampled.emptyStack(),
                    az = downSampled.emptyStack(),
                    zen = downSampled.emptyStack();

            int vecImgDepth = 0;

            int framesPerIteration = framesPerRun(img.dim().height, img.dim().width, img.dim().depth);

            if (framesPerIteration > 1) framesPerIteration = framesPerIteration / 2;
            else if (framesPerIteration <= 0) {
                IJ.error("Your stack has a high depth relative to GPU size. This may cause a crash due to insufficiant GPU memory to process a complete frame.");
                framesPerIteration = 1;
            }

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
                            np.getVectorImg(ui.vfSpacingXY, ui.vfSpacingZ, ui.vfMag, false, ui.overlay ? img.dim() : null),
                            vf
                    );

                if (ui.useCoherence) appendHM(coh, np.getCoherence(ui.tolerance), 0, 1);

                if (ui.saveDatToDir.isPresent())
                    new DatSaver(downSampled, np.stm.getVectors(), handle, ui.saveDatToDir.get(), ui.vfSpacingXY, ui.vfSpacingZ).saveAllVectors();
                
                    

            }

            long endTime = System.currentTimeMillis();

            System.out.println("Execution time: " + (endTime - startTime) + " milliseconds");

            // Removed rawEigenvectors parameter as per instruction to not implement logic here
            results(vf, coh, az, zen, save, ui, img.dim(), vecImgDepth, img);
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
     * Presents the images. This method handles displaying image-based outputs
     * or saving them to file based on the {@code save} parameter. Raw vector
     * data saving (if enabled via UserInput) is handled elsewhere and is not
     * controlled by this method.
     *
     * @param vf The vector field.
     * @param coh The coherence.
     * @param az The azimuthal angles.
     * @param zen The zenith angles.
     * @param save Specifies how to present/save image outputs (fiji, tiff,
     * png).
     * @param ui The user input.
     * @param dims The dimensions.
     * @param vecImgDepth The depth of the vector image.
     * @param myImg The image worked on.
     */
    private static void results(MyImageStack vf, MyImageStack coh, MyImageStack az, MyImageStack zen, Save save, UserInput ui, Dimensions dims, int vecImgDepth, MyImagePlus myImg) {

        if (ui.heatMap) {
            present(az.imp("Azimuthal Angles", dims.depth), save, "N_PIG_images" + File.separatorChar + "Azimuthal");
            if (dims.hasDepth())
                present(zen.imp("Zenith Angles", dims.depth), save, "N_PIG_images" + File.separatorChar + "Zenith");
        }
        if (ui.vectorField) {
            MyImagePlus impVF;
            if (!dims.hasDepth() && ui.overlay)
                impVF = new MyImagePlus("Overlaid Nematic Vectors", myImg.getImageStack(), dims.depth)
                        .overlay(vf, Color.GREEN);
            else impVF = vf.imp("Nematic Vectors", vecImgDepth);
            present(impVF, save, "N_PIG_images" + File.separatorChar + "vectors");
        }

        if (ui.useCoherence) present(coh.imp("Coherence", dims.depth), save, "N_PIG_images" + File.separatorChar + "Coherence");

        // Logic for saving raw vector data (e.g., when save == Save.txt or ui.saveVectorsToFile is true)
        // would be placed here, but is omitted as per instruction.
        // It would typically check ui.saveVectorsToFile and then perform file writing.
    }

    /**
     * Presents the image, either by showing it on Fiji or saving it as a file.
     * This method is specifically for ImagePlus outputs (TIFF, PNG, Fiji
     * display).
     *
     * @param image The image to be presented.
     * @param saveTo The desired save format (fiji, tiff, png).
     * @param filePath The base file path for saving (ignored if displaying in
     * Fiji).
     */
    private static void present(MyImagePlus image, Save saveTo, String filePath) {
        if (saveTo == Save.fiji) {
            image.show();
        } else if (saveTo == Save.tiff || saveTo == Save.png) {
            try {
                Files.createDirectories(Paths.get(filePath));
            } catch (IOException e) {
                System.err.println("Failed to create directory: " + filePath + " - " + e.getMessage());
            }
            image.saveSlices(filePath, saveTo == Save.tiff);
        }
        // Save.txt is intentionally not handled by this method, as it refers to
        // raw data output, not an ImagePlus.
    }

}
