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

        NumericField xyR = new NumericField("Neighborhood xy radius:", 3, gd);
        NumericField zR = null;
        if (hasZ) zR = new NumericField("Neighborhood z radius:", 3, gd);
        BooleanField heatmap = new BooleanField("heatmap", false, gd);
        BooleanField vector = new BooleanField("vector field", true, gd);
        BooleanField coherence = new BooleanField("generate coherence", false, gd);

        NumericField layerDist = null;
        if (hasZ) layerDist = new NumericField("Distance between layers as a multiple of the distance between pixels:", 1, gd);
        NumericField downSample = new NumericField("Downsample Factor XY:", 2, gd);

        gd.showDialog();

        if (gd.wasCanceled())
            throw new UserCanceled();

        NumericField spacing = null, mag = null;
        BooleanField overlay = null;

        
        if (vector.is()) {
            GenericDialog vfDialog = new GenericDialog("Vector Field Parameters.  Be sure downSample > 1.");                        

            if(!hasZ)overlay = new BooleanField("overlay?", false, vfDialog);
            
            spacing = new NumericField("spacing", 8/*downSample.val()*/, vfDialog);
            
            mag = new NumericField("Vector Magnitude:", 6/*downSample.val() - 2*/, vfDialog);
            
            vfDialog.showDialog();
            
            if(!hasZ && overlay.is() && spacing.val() != downSample.val()) throw new RuntimeException("If set to overlay, then spacing must equal downsample size.");
        }
        
        

        return new UserInput(
                new NeighborhoodDim((int) xyR.val(), hasZ ? (int) zR.val() : 1, hasZ ? (int) layerDist.val() : 1),
                heatmap.is(),
                vector.is(),
                coherence.is(),
                vector.is() ? (int) spacing.val() : 0,
                vector.is() ? (int) mag.val() : 0,
                vector.is() && !hasZ ? overlay.is() : false,
                1e-6f,
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
        return new UserInput(nd, true, true, true, spacing, Math.max(spacing - 2, 0), false, 1e-7f, 1);
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
