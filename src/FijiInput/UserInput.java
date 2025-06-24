package FijiInput;

import fijiPlugin.NeighborhoodDim;
import ij.ImagePlus;
import ij.gui.GenericDialog;

/**
 * This class encapsulates user input parameters obtained from a dialog box. It
 * stores settings for neighborhood dimensions, output options (heatmap, vector
 * field, coherence), vector field spacing and magnitude, and tolerance.
 *
 * @author E. Dov Neimand
 */
public class UserInput {//TODO: instead of multiple windows, have one window with active and inactive fields.

    public final static float defaultTolerance = 1e-5f;

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
    public final boolean vectorField, overlay;

    /**
     * A boolean indicating whether to generate coherence information.
     */
    public final boolean useCoherence;

    /**
     * The spacing between vectors in the generated vector field.
     */
    public final int vfSpacing;

    /**
     * The magnitude of the vectors in the generated vector field.
     */
    public final int vfMag;

    /**
     * The tolerance value used in processing.
     */
    public final float tolerance;

    /**
     * 1 in every how many pixels has its structure tensor eiganvector computed
     * computed in the X any Y dimensions.
     */
    public final int downSampleFactorXY;

    /**
     * Constructs a UserInput object with the specified parameters.
     *
     * @param neighborhoodSize The dimensions of the neighborhood.
     * @param heatMap Whether to generate a heatmap.
     * @param vectorField Whether to generate a vector field.
     * @param useCoherence Whether to generate coherence information.
     * @param vfSpacing The spacing between vectors in the vector field.
     * @param vfMag The magnitude of the vectors in the vector field.
     * @param tolerance The tolerance value.
     */
    private UserInput(NeighborhoodDim neighborhoodSize, boolean heatMap, boolean vectorField, boolean useCoherence, int vfSpacing, int vfMag, boolean vfOverlay, float tolerance, int downSampleFactorXY) {
        this.neighborhoodSize = neighborhoodSize;
        this.heatMap = heatMap;
        this.vectorField = vectorField;
        this.useCoherence = useCoherence;
        this.vfMag = vfMag;
        this.tolerance = tolerance;
        this.downSampleFactorXY = downSampleFactorXY;
        this.overlay = vfOverlay;
        this.vfSpacing = overlay ? downSampleFactorXY : vfSpacing;
    }

    /**
     * Creates a UserInput object from a dialog box presented to the user.
     *
     * @param imp The ImagePlus object associated with the input.
     * @return A UserInput object containing the user's input.
     * @throws UserCanceled If the user cancels the dialog box.
     */
    public static UserInput fromDiolog(ImagePlus imp) throws UserCanceled {

        boolean hasZ = imp.getNSlices() > 1;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");

        NumericField xyR = new NumericField("Neighborhood xy radius", 20, gd);
        NumericField zR = null;
        if (hasZ) zR = new NumericField("Neighborhood z radius:", 5, gd);
        BooleanField heatmap = new BooleanField("Heatmap", true, gd);
        BooleanField vector = new BooleanField("Vector field", true, gd);
        BooleanField coherence = new BooleanField("Generate coherence", true, gd);

        NumericField layerDist = null;
        if (hasZ) layerDist = new NumericField("Z axis pixel spacing multiplier", 2, gd);
        NumericField downSample = new NumericField("Downsample factor XY:", 20, gd);

        gd.showDialog();

        if (gd.wasCanceled())
            throw new UserCanceled();

        NumericField spacing = null, mag = null;
        BooleanField overlay = null;

        if (vector.is()) {
            GenericDialog vfDialog = new GenericDialog("Vector Field Parameters.  Be sure downSample > 1.");

            if (!hasZ) overlay = new BooleanField("Overlay", false, vfDialog);

            spacing = new NumericField("Spacing", downSample.val(), vfDialog);

            mag = new NumericField("Vector magnitude:", downSample.val(), vfDialog);

            vfDialog.showDialog();

            if (!hasZ && overlay.is() && spacing.val() != downSample.val()) throw new RuntimeException("If set to overlay, then spacing must equal downsample size.");
        }

        return new UserInput(
                new NeighborhoodDim((int) xyR.val(), hasZ ? (int) zR.val() : 1, hasZ ? (int) layerDist.val() : 1),
                heatmap.is(),
                vector.is(),
                coherence.is(),
                vector.is() ? (int) spacing.val() : 0,
                vector.is() ? (int) mag.val() : 0,
                vector.is() && !hasZ ? overlay.is() : false,
                defaultTolerance,
                (int) downSample.val()
        );
    }

    /**
     * Some default values for testing purposes.
     *
     * @param nd The neighborhood dimensions.
     * @return
     */
    public static UserInput defaultVals(NeighborhoodDim nd) {
        int spacing = 6;
        return new UserInput(nd, false, true, false, spacing, Math.max(spacing - 2, 0), false, defaultTolerance, 1);
    }

    
    
    /**
     * Constructs a {@code UserInput} object by parsing an array of strings.
     * This method is useful for loading parameters from a saved configuration
     * or command-line arguments, where each parameter is provided as a string.
     * The number of elements in the input {@code strings} array is variable
     * and depends on the `depth` of the image and the `vectorField` and `overlay`
     * settings. Parameters related to Z-dimensions or vector field options
     * are omitted from the string array if not applicable.
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
     * <li>If `generate_vector_field` is `true`:
     * <ul>
     * <li>`vector_field_spacing` (int)</li>
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
     * (in which case its value is derived from `vector_field_spacing` and
     * it is omitted from the command-line string).</li>
     * </ol>
     * @param depth The depth (number of Z-slices) of the image stack.
     * @return A new {@code UserInput} object populated with the parsed values.
     * @throws NumberFormatException if any string cannot be parsed into its
     * corresponding numeric or boolean type.
     * @throws ArrayIndexOutOfBoundsException if the {@code strings} array does
     * not contain enough elements for the required parameters based on the logic.
     */
    public static UserInput fromStrings(String[] strings, int depth) {

        int i = 0;

        boolean hasZ = depth > 1;

        int xy = Integer.parseInt(strings[i++]);
        int z = hasZ ? Integer.parseInt(strings[i++]) : 0;
        int distBetweenAdjacentLayer = hasZ ? Integer.parseInt(strings[i++]) : 0;
        boolean heatMap = Boolean.parseBoolean(strings[i++]);
        boolean vectorField = Boolean.parseBoolean(strings[i++]);
        boolean coherence = Boolean.parseBoolean(strings[i++]);
        int vectorFieldSpacing = vectorField ? Integer.parseInt(strings[i++]) : 0;
        int vectorFieldMagnitude = vectorField ? Integer.parseInt(strings[i++]) : 0;
        boolean overlay = vectorField && !hasZ ? Boolean.parseBoolean(strings[i++]) : false;
        int downSample = vectorField && overlay ? vectorFieldSpacing : Integer.parseInt(strings[i++]);

        return new UserInput(
                new NeighborhoodDim(xy, z, distBetweenAdjacentLayer),
                heatMap,
                vectorField,
                coherence,
                vectorFieldSpacing,
                vectorFieldMagnitude,
                overlay,
                defaultTolerance,
                downSample);
    }

    /**
     * Checks if the parameters are valid.
     *
     * @return true if the parameters are valid, false otherwise.
     */
    public boolean validParamaters() {
        return neighborhoodSize.valid() && vfSpacing >= -1 && vfMag >= 0 && tolerance > 0;
    }

    /**
     * The greatest multiple of downSample that is less than origSample.
     *
     * @param origSample An integer.
     * @return The greatest multiple of downSample that is less than origSample.
     */
    public int downSample(int origSample) {
        return (origSample / downSampleFactorXY) * downSampleFactorXY;
    }

    /**
     * Returns a string representation of the UserInput object.
     *
     * @return A string containing the values of the UserInput's fields.
     */
    @Override
    public String toString() {
        return "UserInput{"
                + "neighborhoodSize=" + neighborhoodSize
                + ", heatMap=" + heatMap
                + ", vectorField=" + vectorField
                + ", useCoherence=" + useCoherence
                + ", vfSpacing=" + vfSpacing
                + ", vfMag=" + vfMag
                + ", tolerance=" + tolerance
                + ", downSampleFactorXY=" + downSampleFactorXY
                + '}';
    }
}
