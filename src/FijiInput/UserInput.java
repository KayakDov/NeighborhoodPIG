package FijiInput;

import fijiPlugin.NeighborhoodDim;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.DialogListener; // Import DialogListener
import ij.gui.MultiLineLabel;
import java.awt.AWTEvent;
import java.awt.Button;
import java.awt.Label;
import java.awt.Panel;
import java.awt.TextArea;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * This class encapsulates user input parameters obtained from a dialog box. It
 * stores settings for neighborhood dimensions, output options (heatmap, vector
 * field, coherence), vector field spacing and magnitude, and tolerance.
 *
 * @author E. Dov Neimand
 */
public class UserInput {

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
     * The spacing between vectors in the generated vector field in the xy
     * plane.
     */
    public final int vfSpacingXY;

    /**
     * The spacing between vectors in the generated vector field in the z plane.
     */
    public final int vfSpacingZ;

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
    UserInput(NeighborhoodDim neighborhoodSize, boolean heatMap, boolean vectorField, boolean useCoherence, int vfSpacing, int vfMag, boolean vfOverlay, float tolerance, int downSampleFactorXY, int vfSpacingZ) {
        this.neighborhoodSize = neighborhoodSize;
        this.heatMap = heatMap;
        this.vectorField = vectorField;
        this.useCoherence = useCoherence;
        this.vfMag = vfMag;
        this.tolerance = tolerance;
        this.downSampleFactorXY = downSampleFactorXY;
        this.overlay = vfOverlay;
        this.vfSpacingXY = overlay ? downSampleFactorXY : vfSpacing;
        this.vfSpacingZ = vfSpacingZ;
    }

    /**
     * Some default values for testing purposes.
     *
     * @param nd The neighborhood dimensions.
     * @return
     */
    public static UserInput defaultVals(NeighborhoodDim nd) {
        int spacing = 6;
        return new UserInput(nd, false, true, false, spacing, Math.max(spacing - 2, 0), false, defaultTolerance, 1, spacing);
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
     * <li>If `generate_vector_field` is `true`:
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
     * </ol>
     * @param depth The depth (number of Z-slices) of the image stack.
     * @return A new {@code UserInput} object populated with the parsed values.
     * @throws NumberFormatException if any string cannot be parsed into its
     * corresponding numeric or boolean type.
     * @throws ArrayIndexOutOfBoundsException if the {@code strings} array does
     * not contain enough elements for the required parameters based on the
     * logic.
     */
    public static UserInput fromStrings(String[] strings, int depth) {

        int i = 0;
        System.out.println("--- Parsing User Input from Strings ---");

        boolean hasZ = depth > 1;
        System.out.println("Determined hasZ: " + hasZ);

        int xy = Integer.parseInt(strings[i++]);
        System.out.println("Assigned neighborhood_xy_radius: " + xy + " (from strings[" + (i - 1) + "])");

        int z = hasZ ? Integer.parseInt(strings[i++]) : 0;
        if (hasZ) {
            System.out.println("Assigned neighborhood_z_radius: " + z + " (from strings[" + (i - 1) + "])");
        } else {
            System.out.println("neighborhood_z_radius not applicable (depth <= 1), set to: " + z);
        }

        double distBetweenAdjacentLayer = hasZ ? Double.parseDouble(strings[i++]) : 0;
        if (hasZ) {
            System.out.println("Assigned z_axis_pixel_spacing_multiplier: " + distBetweenAdjacentLayer + " (from strings[" + (i - 1) + "])");
        } else {
            System.out.println("z_axis_pixel_spacing_multiplier not applicable (depth <= 1), set to: " + distBetweenAdjacentLayer);
        }

        boolean heatMap = Boolean.parseBoolean(strings[i++]);
        System.out.println("Assigned generate_heatmap: " + heatMap + " (from strings[" + (i - 1) + "])");

        boolean vectorField = Boolean.parseBoolean(strings[i++]);
        System.out.println("Assigned generate_vector_field: " + vectorField + " (from strings[" + (i - 1) + "])");

        boolean coherence = Boolean.parseBoolean(strings[i++]);
        System.out.println("Assigned generate_coherence: " + coherence + " (from strings[" + (i - 1) + "])");

        int vectorFieldSpacingXY = vectorField ? Integer.parseInt(strings[i++]) : 0;
        if (vectorField) {
            System.out.println("Assigned vector_field_spacing_xy: " + vectorFieldSpacingXY + " (from strings[" + (i - 1) + "])");
        } else {
            System.out.println("vector_field_spacing_xy not applicable (generate_vector_field is false), set to: " + vectorFieldSpacingXY);
        }

        int vectorFieldSpacingZ = hasZ && vectorField ? Integer.parseInt(strings[i++]) : 0;
        if (hasZ && vectorField) {
            System.out.println("Assigned vector_field_spacing_z: " + vectorFieldSpacingZ + " (from strings[" + (i - 1) + "])");
        } else {
            System.out.println("vector_field_spacing_z not applicable (depth <= 1 or generate_vector_field is false), set to: " + vectorFieldSpacingZ);
        }

        int vectorFieldMagnitude = vectorField ? Integer.parseInt(strings[i++]) : 0;
        if (vectorField) {
            System.out.println("Assigned vector_field_magnitude: " + vectorFieldMagnitude + " (from strings[" + (i - 1) + "])");
        } else {
            System.out.println("vector_field_magnitude not applicable (generate_vector_field is false), set to: " + vectorFieldMagnitude);
        }

        boolean overlay = vectorField && !hasZ ? Boolean.parseBoolean(strings[i++]) : false;
        if (vectorField && !hasZ) {
            System.out.println("Assigned overlay_vector_field: " + overlay + " (from strings[" + (i - 1) + "])");
        } else {
            System.out.println("overlay_vector_field not applicable (generate_vector_field is false or depth > 1), set to: " + overlay);
        }

        // Default tolerance value for UserInput constructor, not parsed from strings
        double defaultTolerance = 0.0;

        int downSample;
        if (overlay) {
            downSample = vectorFieldSpacingXY;
            System.out.println("Assigned downsample_factor_xy: " + downSample + " (derived from vector_field_spacing_xy due to overlay being true)");
        } else {
            downSample = Integer.parseInt(strings[i++]);
            System.out.println("Assigned downsample_factor_xy: " + downSample + " (from strings[" + (i - 1) + "])");
        }
        System.out.println("--- Finished Parsing User Input ---");

        return new UserInput(
                new NeighborhoodDim(xy, z, distBetweenAdjacentLayer),
                heatMap,
                vectorField,
                coherence,
                vectorFieldSpacingXY,
                vectorFieldMagnitude,
                overlay,
                (float)defaultTolerance, // This is a default value and not parsed from the string array
                downSample,
                vectorFieldSpacingZ
        );
    }

    /**
     * Checks if the parameters are valid.
     *
     * @return true if the parameters are valid, false otherwise.
     */
    public boolean validParamaters() {
        return neighborhoodSize.valid() && vfSpacingXY >= 0 && vfSpacingZ >= 0 && vfMag >= 0 && tolerance >= 0;
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
                + ", vfSpacing=" + vfSpacingXY
                + ", vfMag=" + vfMag
                + ", tolerance=" + tolerance
                + ", downSampleFactorXY=" + downSampleFactorXY
                + '}';
    }
}
