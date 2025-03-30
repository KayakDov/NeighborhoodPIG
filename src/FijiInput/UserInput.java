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
    public final boolean vectorField;

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
    private UserInput(NeighborhoodDim neighborhoodSize, boolean heatMap, boolean vectorField, boolean useCoherence, int vfSpacing, int vfMag, float tolerance) {
        this.neighborhoodSize = neighborhoodSize;
        this.heatMap = heatMap;
        this.vectorField = vectorField;
        this.useCoherence = useCoherence;
        this.vfSpacing = vfSpacing;
        this.vfMag = vfMag;
        this.tolerance = tolerance;
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
        NumericField layerDist = new NumericField("Distance between layers as a multiple of the distance between pixels:", 1, gd);
        NumericField xyR = new NumericField("Neighborhood xy radius:", 30, gd);
        NumericField zR = null;
        if (hasZ) zR = new NumericField("Neighborhood z radius:", 1, gd);
        BooleanField heatmap = new BooleanField("heatmap", true, gd);
        BooleanField vector = new BooleanField("vector field", false, gd);
        BooleanField coherence = new BooleanField("generate coherence", false, gd);
        
        gd.showDialog();
        
        if (gd.wasCanceled()) throw new UserCanceled();

        NumericField spacing = null, mag = null;
        
        if (vector.is()) {
            GenericDialog vfDialog = new GenericDialog("Vector Field Parameters");
            spacing = new NumericField("Vector Field Spacing:", 8, vfDialog);
            mag = new NumericField("Vector Field Magnitude:", 6, vfDialog);
            vfDialog.showDialog();
        }

        
        return new UserInput(
                new NeighborhoodDim((int) xyR.getVal(), hasZ ?  (int)zR.getVal(): 0, (int)layerDist.getVal()),
                heatmap.is(),
                vector.is(),
                coherence.is(),
                vector.is()?(int)spacing.getVal():0,
                vector.is()?(int)mag.getVal():0,
                1
        );
    }

    /**
     * Checks if the parameters are valid.
     *
     * @return true if the parameters are valid, false otherwise.
     */
    public boolean validParamaters() {
        return neighborhoodSize.valid() && vfSpacing >= 0 && vfMag >= 0 && tolerance > 0;
    }
}
