package FijiInput;

import FijiInput.field.VF;
import fijiPlugin.Dimensions;
import fijiPlugin.NeighborhoodDim;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageConverter;
import imageWork.MyImagePlus;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Optional;

/**
 * This class encapsulates user input parameters obtained from a dialog box. It
 * stores settings for neighborhood dimensions, output options (heatmap, vector
 * field, coherence), vector field spacing and magnitude, and tolerance.
 *
 * @author E. Dov Neimand
 */
public class UsrInput {

    public final static double defaultTolerance = 1e-5f;

    public final MyImagePlus img;
    public final Dimensions dim;

    /**
     * The dimensions of the neighborhood used for processing.
     */
    public final NeighborhoodDim neighborhoodSize;

    /**
     * A boolean indicating whether to generate a heatmap.
     */
    public final boolean heatMap;

    /**
     * A boolean indicating whether to generate a vector field.
     */
    public final VF vectorField;

    /**
     * Should the vector field overlay the image?
     */
    public final Optional<Boolean> overlay;

    /**
     * A boolean indicating whether to generate coherence information.
     */
    public final boolean coherence;

    /**
     * A boolean indicating whether to save the computed vectors to a file.
     */
    public final Optional<Path> saveDatToDir;

    /**
     * The spacing between vectors in the generated vector field in the xy
     * plane.
     */
    public final Optional<Integer> spacingXY;

    /**
     * The spacing between vectors in the generated vector field in the z plane.
     */
    public final Optional<Integer> spacingZ;

    /**
     * The magnitude of the vectors in the generated vector field.
     */
    public final Optional<Integer> vfMag;

    /**
     * The tolerance value used in processing.
     */
    public final double tolerance;

    /**
     * 1 in every how many pixels has its structure tensor eiganvector computed
     * computed in the X any Y dimensions.
     */
    public final int downSampleFactorXY;

    /**
     * 1 in every how many pixels has its structure tensor eiganvector computed
     * computed in the Z dimension.
     */
    public final Optional<Integer> downSampleFactorZ;

    /**
     * Constructs a UserInput object with the specified parameters.
     *
     * @param img The image to be analyzed.
     * @param neighborhoodSize The dimensions of the neighborhood.
     * @param heatMap Whether to generate a heatmap.
     * @param vectorField Whether to generate a vector field.
     * @param useCoherence Whether to generate coherence information.
     * @param saveDatToDir Whether to save the computed vectors to a file. Set
     * this to null if the vectors should not ve saved to a .dat file.
     * @param vfOverlay true if the vector field should overlay the image, false
     * if it should not, absent if this is not a 2d image with a request for a
     * vector field.
     * @param vfSpacingXY The spacing between vectors in the vector field.
     * @param vfMag The magnitude of the vectors in the vector field.
     * @param tolerance The tolerance value.
     * @param downSampleFactorXY The downsample factor for XY dimensions.
     * @param vfSpacingZ The spacing between vectors in the Z dimension.
     * @param downSampleFactorZ The downsample factor for Z dimension.
     */
    public UsrInput(ImagePlus img, NeighborhoodDim neighborhoodSize, boolean heatMap, VF vectorField,
            boolean useCoherence, Optional<Path> saveDatToDir, Optional<Boolean> vfOverlay,
            Optional<Integer> vfMag, Optional<Integer> vfSpacingXY, Optional<Integer> vfSpacingZ,
            int downSampleFactorXY, Optional<Integer> downSampleFactorZ, double tolerance) {
        this.neighborhoodSize = neighborhoodSize;
        this.heatMap = heatMap;
        this.vectorField = vectorField;
        this.coherence = useCoherence;
        this.saveDatToDir = saveDatToDir;
        this.vfMag = vfMag;
        this.tolerance = tolerance;
        this.downSampleFactorXY = downSampleFactorXY;
        this.overlay = vfOverlay;
        this.spacingXY = vfSpacingXY;
        this.spacingZ = vfSpacingZ;
        this.downSampleFactorZ = downSampleFactorZ;
        
        if(!validImage(img)) IJ.error("There's something wrong with your image.");
        
        this.img = new MyImagePlus(img).crop(
                downSampleXY(img.getHeight()),
                downSampleXY(img.getWidth()),
                downSampleZ(img.getNSlices())
        );
        img.setOpenAsHyperStack(true);
        dim = this.img.dim().downSample(null, downSampleFactorXY, downSampleFactorZ.orElse(1));;
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
     * Constructs a {@code UserInput} object by parsing an array of strings.
     * This method is useful for loading parameters from a saved configuration
     * or command-line arguments, where each parameter is provided as a string.
     * The number of elements in the input {@code strings} array is variable and
     * depends on the `depth` of the image and the `vectorField` and `overlay`
     * settings. Parameters related to Z-dimensions or vector field options are
     * omitted from the string array if not applicable.
     *
     * @param strings An array of strings containing the user input parameters
     * in a specific order. The presence of parameters depends on the `depth`
     * and other boolean flags. The order is:
     * <ol>
     * <li>`neighborhood_xy_radius` (int)</li>
     * <li>If image has Z-depth (`depth > 1`):
     * <ul>
     * <li>`neighborhood_z_radius` (int)</li>
     * <li>`z_axis_pixel_spacing_multiplier` (int)</li>
     * </ul>
     * </li>
     * <li>`generate_heatmap` (boolean)</li>
     * <li>`generate_vector_field` (boolean)</li>
     * <li>`generate_coherence` (boolean)</li>
     * <li>`save_vectors_to_file` (boolean)</li> * <li>If
     * `generate_vector_field` is `true`:
     * <ul>
     * <li>`vector_field_spacing_xy` (int)</li>
     * <li>If image has Z-depth (`depth > 1`):
     * <ul>
     * <li>`vector_field_spacing_z` (int)</li>
     * </ul>
     * </li>
     * <li>`vector_field_magnitude` (int)</li>
     * <li>If image has no Z-depth (`depth == 1`):
     * <ul>
     * <li>`overlay_vector_field` (boolean)</li>
     * </ul>
     * </li>
     * </ul>
     * </li>
     * <li>`downsample_factor_xy` (int) - This parameter is present *unless*
     * `generate_vector_field` is `true` AND `overlay_vector_field` is `true`
     * (in which case its value is derived from `vector_field_spacing` and it is
     * omitted from the command-line string).</li>
     * <li>`downsample_factor_z` (int)</li>
     * </ol>
     * @param depth The depth (number of Z-slices) of the image stack.
     * @param img the image
     * @return A new {@code UserInput} object populated with the parsed values.
     * @throws NumberFormatException if any string cannot be parsed into its
     * corresponding numeric or boolean type.
     * @throws ArrayIndexOutOfBoundsException if the {@code strings} array does
     * not contain enough elements for the required parameters based on the
     * logic.
     */
    public static UsrInput fromStrings(String[] strings, ImagePlus img) {

        System.out.println("FijiInput.UserInput.fromStrings()" + Arrays.toString(strings));

        System.out.println("--- Parsing User Input from Strings ---");

        boolean hasZ = img.getNSlices() > 1;
        System.out.println("Determined hasZ: " + hasZ);

        StringIter si = new StringIter(strings);

        int xyR = ParseInt.from(si, "xy neighborhood radius").get();

        Optional<Integer> zR = ParseInt.from(si, "z neighborhood radius", hasZ);

        Optional<Double> distBetweenAdjacentLayer = ParseReal.from(si, "distance factor for adjacent layers", hasZ);

        boolean heatMap = ParseBool.from(si, "make heatmap").get();

        VF vectorField = ParseEnum.from(si, VF.class, "make vector field").get();

        Boolean coherence = ParseBool.from(si, "make coherence").get();

        Optional<Path> saveVectors = ParsePath.from(si, "save vectors to dat file");

        Optional<Integer> vectorFieldSpacingXY = ParseInt.from(si, "xy spacing", vectorField.is());

        Optional<Integer> vectorFieldSpacingZ = ParseInt.from(si, "z spacing", hasZ && vectorField.is());

        Optional<Integer> vectorFieldMagnitude = ParseInt.from(si, "vector magnitude", vectorField.is());

        Optional<Boolean> overlay = ParseBool.from(si, "overlay vector field", vectorField.is() && !hasZ);

        int downSampleXY = ParseInt.from(si, "down sample xy").get();

        Optional<Integer> downSampleZ = ParseInt.from(si, "down sample z", hasZ);

        System.out.println("--- Finished Parsing User Input ---");

        return new UsrInput(
                img,
                new NeighborhoodDim(xyR, zR, distBetweenAdjacentLayer),
                heatMap,
                vectorField,
                coherence,
                saveVectors,
                overlay,
                vectorFieldMagnitude,
                vectorFieldSpacingXY,
                vectorFieldSpacingZ, // This is a default value and not parsed from the string array
                downSampleXY,
                downSampleZ,
                defaultTolerance
        );
    }

    /**
     * Checks if the parameters are valid.
     *
     * @return true if the parameters are valid, false otherwise.
     */
    public boolean validParamaters() {
        
        return neighborhoodSize.valid()
                && spacingXY.orElse(2) > 1
                && spacingZ.orElse(2) > 1
                && vfMag.orElse(2) > 1
                && tolerance > 0
                && (!overlay.orElse(false)
                || (downSampleFactorXY == spacingXY.get()
                && downSampleFactorZ.orElse(0) == spacingZ.orElse(0)))
                && downSampleFactorXY >= 1
                && downSampleFactorZ.orElse(2) >= 1;
    }

    /**
     * The greatest multiple of downSample that is less than origSample.
     *
     * @param origSample An integer.
     * @return The greatest multiple of downSample that is less than origSample.
     */
    public final int downSampleXY(int origSample) {
        return (origSample / downSampleFactorXY) * downSampleFactorXY;
    }

    /**
     * The greatest multiple of downSampleFactorZ that is less than origSample.
     *
     * @param origSample An integer.
     * @return The greatest multiple of downSampleFactorZ that is less than
     * origSample.
     */
    public final int downSampleZ(int origSample) {
        return (origSample / downSampleFactorZ.orElse(1)) * downSampleFactorZ.orElse(1);
    }

    @Override
    public String toString() {
        return "UserInput {\n"
                + "  neighborhoodSize           = " + neighborhoodSize + ",\n"
                + "  heatMap                    = " + heatMap + ",\n"
                + "  vectorField                = " + vectorField + ",\n"
                + "  overlay                   = " + overlay + ",\n"
                + "  useCoherence               = " + coherence + ",\n"
                + "  saveVectorsToFile          = " + saveDatToDir + ",\n" // NEW TOSTRING ENTRY
                + "  vfSpacingXY                = " + spacingXY + ",\n"
                + "  vfSpacingZ                 = " + spacingZ + ",\n"
                + "  vfMag                      = " + vfMag + ",\n"
                + "  tolerance                  = " + tolerance + ",\n"
                + "  downSampleFactorXY         = " + downSampleFactorXY + ",\n"
                + "  downSampleFactorZ          = " + downSampleFactorZ + "\n"
                + '}';
    }

}
