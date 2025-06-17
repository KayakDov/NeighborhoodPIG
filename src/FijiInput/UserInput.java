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
        this.vfSpacing = vfSpacing;
        this.vfMag = vfMag;
        this.tolerance = tolerance;
        this.downSampleFactorXY = downSampleFactorXY;
        this.overlay = vfOverlay;
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

        NumericField xyR = new NumericField("Neighborhood xy radius:", 15, gd);
        NumericField zR = null;
        if (hasZ) zR = new NumericField("Neighborhood z radius:", 1, gd);
        BooleanField heatmap = new BooleanField("heatmap", true, gd);
        BooleanField vector = new BooleanField("vector field", false, gd);
        BooleanField coherence = new BooleanField("generate coherence", true, gd);

        NumericField layerDist = null;
        if (hasZ) layerDist = new NumericField("Distance between layers as a multiple of the distance between pixels:", 1, gd);
        NumericField downSample = new NumericField("Downsample Factor XY:", 1, gd);

        gd.showDialog();

        if (gd.wasCanceled())
            throw new UserCanceled();

        NumericField spacing = null, mag = null;
        BooleanField overlay = null;

        if (vector.is()) {
            GenericDialog vfDialog = new GenericDialog("Vector Field Parameters.  Be sure downSample > 1.");

            if (!hasZ) overlay = new BooleanField("overlay?", false, vfDialog);

            spacing = new NumericField("spacing", downSample.val(), vfDialog);

            mag = new NumericField("Vector Magnitude:", downSample.val() - 2, vfDialog);

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
        return new UserInput(nd, true, false, true, spacing, Math.max(spacing - 2, 0), false, defaultTolerance, 1);
    }

    /**
     * Constructs a {@code UserInput} object by parsing an array of strings.
     * This method is useful for loading parameters from a saved configuration
     * or command-line arguments, where each parameter is provided as a string.
     * The order of elements in the input {@code string} array is crucial and
     * must match the expected parsing sequence.
     *
     * @param string An array of strings containing the user input parameters in
     * the following order.
     * <ol>
     * <li>XY radius of the neighborhood (int)</li>
     * <li>Z radius of the neighborhood. (int) Omit if the image has no
     * depth.</li>
     * <li>Distance between adjacent layers (int) Omit if the image has no
     * depth.</li>
     * <li>Whether to generate a heatmap (boolean)</li>
     * <li>Whether to generate a vector field (boolean)</li>
     * <li>Whether to generate coherence information (boolean)</li>
     * <li>Vector field spacing (int) Omit if the image has no depth.</li>
     * <li>Vector field magnitude (int) Omit if the image has no depth.</li>
     * <li>Whether to overlay the vector field (boolean) - Omit if vector field
     * is false or the image has more than 1 slice</li>
     * <li>Downsample factor XY (int) - Only include if vector field is true and
     * overlay is true, this value is taken from vector field spacing;
     * otherwise, it's parsed directly.</li>
     * </ol>
     * @param imp The {@code ImagePlus} object, used to determine if the image
     * has multiple slices (Z-dimension) which affects the parsing of the
     * overlay parameter.
     * @return A new {@code UserInput} object populated with the parsed values.
     * @throws NumberFormatException if any string cannot be parsed into its
     * corresponding numeric or boolean type.
     * @throws ArrayIndexOutOfBoundsException if the {@code string} array does
     * not contain enough elements for the required parameters.
     */
    public static UserInput fromStrings(String string, ImagePlus imp) {
        
        int i = 0;

        String[] strings = string.split(" ");
        
        boolean hasZ = imp.getNSlices() > 1;

        int xy = Integer.parseInt(strings[i++]);
        int z = hasZ ? Integer.parseInt(strings[i++]) : 0;
        int distBetweenAdjacentLayer = hasZ ? Integer.parseInt(strings[i++]) : 0;
        boolean heatMap = Boolean.parseBoolean(strings[i++]);
        boolean vectorField = Boolean.parseBoolean(strings[i++]);
        boolean coherence = Boolean.parseBoolean(strings[i++]);
        int vectorFieldSpacing = vectorField ? Integer.parseInt(strings[i++]) : 0;
        int vectorFieldMagnitude = vectorField ? Integer.parseInt(strings[i++]) : 0;
        boolean overlay = vectorField && imp.getNSlices() == 1 ? Boolean.parseBoolean(strings[i++]) : false;
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
        return neighborhoodSize.valid() && vfSpacing >= 0 && vfMag >= 0 && tolerance > 0;
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
