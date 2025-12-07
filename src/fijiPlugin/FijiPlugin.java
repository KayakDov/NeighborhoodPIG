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
            if(UsrDialog.getIJFrontImage() != null) {
                UsrDialog ud = new UsrDialog();
            }
            else  IJ.error("Your must select an image.");
        } else
            new Launcher(UsrInput.fromStrings(string.split(" "), UsrDialog.getIJFrontImage()), Launcher.Save.png).run();
        
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

        System.out.println("fijiPlugin.FijiPlugin.main()before \n" + GPU.memory());

        if (args.length == 0) {
            args = defaultArgs();
        }

        FijiPlugin fp = new FijiPlugin();

        int depth = Integer.parseInt(args[1]);

        ImagePlus imp = ProcessImage.getImagePlus(args[0], depth);

        new Launcher(UsrInput.fromStrings(Arrays.copyOfRange(args, 3, args.length), imp), Launcher.Save.valueOf(args[2])).run();        

    }

    private static String[] defaultArgs() {
        String imagePath = "/home/dov/Downloads/OneDrive_1_11-26-2025";
        int depth = 34;

//        String imagePath = "images/input/SingleTest/";
//        int depth = 1;
        int xyR = 5;
        int zR = 5;
        double zDist = 1;
        boolean hasHeatMap = true;
        VF hasVF = VF.Color;
        boolean hasCoherence = true;
        String saveVectors = "false";
        int vfSpacingXY = 15;
        int vfSpacingZ = 5;
        int mag = 15;
        boolean overlay = false;
        int downSampleXY = 10;
        int downSampleZ = 3;

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

}
