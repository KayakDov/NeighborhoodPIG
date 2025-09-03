package fijiPlugin;

import FijiInput.NoImage;
import FijiInput.UserCanceled;
import FijiInput.UsrDialog;
import FijiInput.UsrInput;
import FijiInput.field.VF;
import JCudaWrapper.array.Array;
import JCudaWrapper.resourceManagement.GPU;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Toolbar;
import ij.plugin.PlugIn;
import imageWork.TxtSaver;
import imageWork.HeatMapCreator;
import imageWork.MyImagePlus;
import imageWork.MyImageStack;
import imageWork.ProcessImage;
import imageWork.vectors.VectorImg;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.SwingUtilities;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn {

    public MyImageStack vf, coh, az, zen;
    public final ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private UsrInput ui;

    /**
     * initializes the stacks
     *
     * @param ui The user's inputs.
     */
    private void initStacks() {
        vf = VectorImg.space(ui.dim, ui.spacingXY.orElse(1), ui.spacingZ.orElse(1), ui.vfMag.orElse(1), ui.overlay.orElse(false) ? ui.img.dim() : null).emptyStack();
        coh = ui.dim.emptyStack();
        az = ui.dim.emptyStack();
        zen = ui.dim.emptyStack();
    }

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
     * The maximum number of frames that can be processed at once. TODO: this
     * should take downsampling into account!
     *
     * @return The maximum number of frames that can be processed at once.
     */
    public int framesPerRun() {

        long freeMemory = GPU.freeMemory();

        if (freeMemory == 0)
            throw new RuntimeException("There is no free GPU memory.");

        long voxlesPerFrame = (long) ui.dim.height * ui.dim.width * ui.dim.depth;

        int framesPerRun = (int) ((freeMemory / voxlesPerFrame) / (Sizeof.DOUBLE * (ui.dim.depth > 1 ? 6 : 3) + Sizeof.FLOAT * (ui.dim.depth > 1 ? 3 : 2)));

        if (framesPerRun > 1)
            return framesPerRun / 2;
        else if (framesPerRun <= 0)
            IJ.error("Your stack has a high depth relative to GPU size. This may cause a crash due to insufficiant GPU memory to process a complete frame.");

        return 1;

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
     * {@link UsrInput#fromStrings(String[], int) UserInput.fromStrings} method.
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

        if (string.length() == 0) {
            try {
                ui = new UsrDialog().getUserInput();
            } catch (UserCanceled ex) {
                System.out.println("fijiPlugin.FijiPlugin.run() User canceled dialog.");
                return;
            } catch(NoImage nie){
                IJ.error("Your must select an image.");
                return;
            }
        } else {
            ui = UsrInput.fromStrings(string.split(" "), UsrDialog.getIJFrontImage());
        }
        run(Save.fiji);
    }

    /**
     * Loads image J. This should be called if this is not being run from Fiji.
     */
    public static void loadImageJ() {
        ImageJ ij = IJ.getInstance();
        if (ij == null)
            ij = new ImageJ(ImageJ.NO_SHOW); // Start ImageJ without showing the main window initially

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
     * {@link FijiInput.UsrInput#fromStrings(String[], int) UserInput.fromStrings}
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
    public static void main(String[] args) {//        loadImageJ();

        if (args.length == 0)
            args = defaultArgs();

        FijiPlugin fp = new FijiPlugin();

        int depth = Integer.parseInt(args[1]);

        ImagePlus imp = ProcessImage.getImagePlus(args[0], depth);

        fp.ui = UsrInput.fromStrings(
                Arrays.copyOfRange(args, 3, args.length),
                imp
        );

        fp.run(Save.valueOf(args[2]));

    }

    private static String[] defaultArgs() {//TODO: Should be able to save data and view a file with the same run command.
        String imagePath = "images/input/AngleCyl/";
        int depth = 25;

//        String imagePath = "images/input/SingleTest/";
//        int depth = 1;
        int xyR = 5;
        int zR = 5;
        double zDist = 1;
        boolean hasHeatMap = true;
        VF hasVF = VF.Color;
        boolean hasCoherence = false;
        String saveVectors = "false";
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
            "" + xyR,
            "" + zR,
            "" + zDist,
            "" + hasHeatMap,
            "" + hasVF,
            "" + hasCoherence,
            "" + saveVectors, // This flag would be parsed by UserInput.fromStrings
            "" + vfSpacingXY,
            "" + vfSpacingZ,
            "" + mag,
            //            "" + overlay,
            "" + downSampleXY,
            "" + downSampleZ
        };

    }

    /**
     * Allows for code to be run both from the main and from run.
     *
     * @param save Where to save the file to.
     */
    public void run(Save save) {
        if (!ui.validParamaters())
            throw new RuntimeException("fijiPlugin.FijiPlugin.run() Invalid Parameters!");

        try {

            initStacks();

            int vecImgDepth = 0;

            int framesPerIteration = framesPerRun();

            long startTime = System.currentTimeMillis();

            for (int i = 0; i < ui.img.getNFrames(); i += framesPerIteration)
                try (Handle handle = new Handle(); NeighborhoodPIG np = new NeighborhoodPIG(handle, ui.img.subset(i, framesPerIteration), ui)) {
                vecImgDepth = processNPIGResults(ui, handle, np);
            }

            awaitThreadTermination();
            printTime(startTime);

            results(save, ui, vecImgDepth);
        } catch (Exception ex) {
            System.out.println("fijiPlugin.FijiPlugin.run() " + ui.toString() + " " + ex.toString());
            throw ex;
        }
        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak.");
    }

    /**
     * Takes the results from running NeighborhoodPIG and loads them into
     * imageStacks or saved files based on the user's input. This method
     * orchestrates the creation of various output images such as heatmaps for
     * azimuthal and zenith angles, vector fields, and coherence maps. It also
     * handles the saving of raw vector data if specified by the user.
     *
     * @param ui The {@link UsrInput} object containing all the user-defined
     * parameters and preferences for output generation (e.g., whether to
     * generate heatmaps, vector fields, coherence, or save raw data).
     * @param fp The {@link FijiPlugin} instance, which provides access to
     * shared resources like dimensions ({@link FijiPlugin#dim}), image stacks
     * for different outputs (e.g., {@link FijiPlugin#az},
     * {@link FijiPlugin#zen}, {@link FijiPlugin#vf},
     * {@link FijiPlugin#coh}), and the executor service ({@link FijiPlugin#es})
     * for parallel processing.
     * @param handle A {@link Handle} object, typically representing a GPU
     * device handle or context, used for managing GPU memory and kernel
     * execution within the current processing iteration.
     * @param np The {@link NeighborhoodPIG} object, which contains the computed
     * results for the current image frame(s), including coherence, vector
     * types, and methods to retrieve azimuthal angles, zenith angles, and
     * vector images.
     * @return The depth of the generated vector image if a vector field was
     * created ({@code ui.vectorField.is()} is true); otherwise, returns 0. This
     * value is used subsequently for displaying or saving vector field results
     * correctly.
     */
    private int processNPIGResults(UsrInput ui, Handle handle, NeighborhoodPIG np) {

        if (ui.heatMap) {

            appendHM(az, np.getAzimuthalAnglesHeatMap(ui.tolerance), 0, (float) Math.PI, es);
            if (ui.img.dim().hasDepth())
                appendHM(zen, np.getZenithAnglesHeatMap(false, 0.01), 0, (float) Math.PI, es);
        }

        if (ui.vectorField.is()) {
            return appendVF(
                    ui,
                    np.getVectorImg(
                            ui.spacingXY.get(),
                            ui.spacingZ.orElse(1),
                            ui.vfMag.get(),
                            false,
                            ui.overlay.orElse(false) ? ui.img.dim() : null,
                            ui.vectorField == VF.Color
                    ),
                    vf,
                    es
            );
        }

        if (ui.coherence)
            appendHM(coh, np.getCoherenceHeatMap(ui.tolerance), 0, 1, es);

        if (ui.saveDatToDir.isPresent())
            new TxtSaver(ui.dim, np.stm.getVectors(), handle, ui.saveDatToDir.get(), ui.spacingXY.orElse(1), ui.spacingZ.orElse(1), np.stm.coherence, ui.tolerance).saveAllVectors();

        return 0;
    }

    /**
     * prints the amount of time since the clock started.
     *
     * @param startTime
     */
    private static void printTime(long startTime) {
        long endTime = System.currentTimeMillis();
        System.out.println("Execution time: " + (endTime - startTime) + " milliseconds");
    }

    /**
     * Waits for all open threads to close.
     *
     * @param threads All open threads.
     */
    private void awaitThreadTermination() {
        es.shutdown();

        try {
            es.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException ex) {
            Logger.getLogger(FijiPlugin.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Appends to the stack from the vector field.
     *
     * @param ui The user input.
     * @param vecImg The vector image to be added to the stack.
     * @param vf The stack to be appended to.
     * @param es The executor service.
     * @return The depth of the vector image.
     */
    public static int appendVF(UsrInput ui, VectorImg vecImg, MyImageStack vf, ExecutorService es) {
        vf.concat(vecImg.imgStack(es));
        return vecImg.getOutputDimensions().depth;
    }

    /**
     * Appends the heatmap to the stack.
     *
     * @param addTo The stack to have the heatmap's stack added onto it.
     * @param add The heatmap whose stack is to be appended.
     * @param min The minimum value in the stack.
     * @param max The maximum value in the stack.
     * @param es Manages the cpu threads.
     */
    public static void appendHM(MyImageStack addTo, HeatMapCreator add, float min, float max, ExecutorService es) {

        addTo.concat(add.getStack(min, max, es));

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
    private void results(Save save, UsrInput ui, int vecImgDepth) {

        if (ui.heatMap) {
            present(az.getImagePlus("Azimuthal Angles", ui.dim.depth), save, "N_PIG_images" + File.separatorChar + "Azimuthal");
            if (ui.dim.hasDepth())
                present(zen.getImagePlus("Zenith Angles", ui.dim.depth), save, "N_PIG_images" + File.separatorChar + "Zenith");
        }
        if (ui.vectorField.is())
            present(
                    !ui.dim.hasDepth() && ui.overlay.orElse(false)
                    ? new MyImagePlus("Overlaid Nematic Vectors", ui.img.getImageStack(), ui.dim.depth).overlay(vf, Color.GREEN)
                    : vf.getImagePlus("Nematic Vectors", vecImgDepth),
                    save,
                    "N_PIG_images" + File.separatorChar + "vectors"
            );

        if (ui.coherence)
            present(coh.getImagePlus("Coherence", ui.dim.depth), save, "N_PIG_images" + File.separatorChar + "Coherence");
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

            image.setOpenAsHyperStack(true);

            image.show();

        } else if (saveTo == Save.tiff || saveTo == Save.png) {
            try {
                Files.createDirectories(Paths.get(filePath));
            } catch (IOException e) {
                System.err.println("Failed to create directory: " + filePath + " - " + e.getMessage());
            }
            image.saveSlices(filePath, saveTo == Save.tiff);
        }

    }

}
