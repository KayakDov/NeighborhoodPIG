package FijiInput;

import static FijiInput.UserInput.defaultTolerance;
import fijiPlugin.NeighborhoodDim; // Assuming NeighborhoodDim is in fijiPlugin package
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.DialogListener;
import java.awt.AWTEvent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * This class is responsible for displaying a graphical user interface (GUI)
 * using ImageJ's GenericDialog to collect input parameters from the user. It
 * manages the lifecycle of the dialog, user interactions, and then constructs
 * an immutable {@link UserInput} object from the collected data.
 *
 * @author E. Dov Neimand
 */
public class UserDialog {

    private UserInput ui;

    /**
     * Creates a UserInput object from a dialog box presented to the user.
     *
     * @param imp The ImagePlus object associated with the input.
     * @throws UserCanceled If the user cancels the dialog box.
     */
    public UserDialog(ImagePlus imp) throws UserCanceled {
        boolean hasZ = imp.getNSlices() > 1;

        GenericDialog gd = new GenericDialog("NeighborhoodPIG Parameters");
        HelpDialog hf = addHelpButton(gd);

        NumericField downSample = new NumericField("Downsample factor XY:", 1, gd, 0,
                "Determines how many pixels are skipped (1 = every pixel, 2 = every other). \nIncreased values improve memory & time performance.",
                hf);

        BooleanField heatmap = new BooleanField("Heatmap", true, gd,
                "Generates a heatmap visualizing image orientations.",
                hf);

        NumericField xyR = new NumericField("Neighborhood xy radius", 5, gd, 0,
                "Radius (pixels) of the square neighborhood in the XY plane for structure tensor calculation.",
                hf);
        NumericField zR = hasZ ? new NumericField("Neighborhood z radius:", 5, gd, 0,
                "Radius (pixels) of the cube neighborhood in the Z-direction for 3D stacks.",
                hf) : null;
        NumericField layerDist = hasZ ? new NumericField("Z axis pixel spacing multiplier", 1, gd, 1,
                "Factor for anisotropic Z-spacing (e.g., 1.5 if Z-distance is 1.5 x XY pixel size).",
                hf) : null;

        BooleanField coherence = new BooleanField("Generate coherence", false, gd,
                "Computes and displays a heatmap representing pixel coherence.",
                hf);

        BooleanField vectorField = new BooleanField("Vector field", false, gd,
                "Displays vectors representing orientations.",
                hf);
        BooleanField overlay = hasZ ? null : new BooleanField("Overlay Vector Field", false, gd,
                "Overlays the vector field directly on the original image.",
                hf);

        NumericField spacingXY = new NumericField("Vector Field Spacing XY:", 0, gd, 0,
                "Distance (pixels) between vectors in the xy plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf);

        NumericField spacingZ = hasZ ? new NumericField("Vector Field Spacing Z:", 0, gd, 0,
                "Distance (pixels) between vectors in the xy plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf) : null;

        NumericField mag = new NumericField("Vector Field Magnitude:", 0, gd, 0,
                "Visual length (pixels) of the displayed vectors.",
                hf);

        gd.addDialogListener(new DialogListener() {
            @Override
            public boolean dialogItemChanged(GenericDialog gd, AWTEvent e) {

                spacingXY.setEnabled(vectorField.is());

                mag.setEnabled(vectorField.is());

                if (hasZ) 
                    spacingZ.setEnabled(vectorField.is());                    
                else {
                    overlay.setEnabled(vectorField.is());

                    if (vectorField.is() && overlay.is())
                        spacingXY.val(downSample.val()).setEnabled(false);

                    if (!vectorField.is()) overlay.is(false);

                }
                if (vectorField.is()) {
                    if (downSample.val() == 1 && downSampelOrig){
                        downSample.val((int) xyR.val());
                        downSampelOrig = false;
                    }
                    if(spacingXY.val() == 0) spacingXY.val(xyR.val());
                    if(hasZ && spacingZ.val() == 0) spacingZ.val(xyR.val());
                    if(mag.val() == 0) mag.val(xyR.val());
                } else {
                    if (spacingXY.val() != 0) spacingXY.val(0);
                    if (mag.val() != 0) mag.val(0);
                    if (hasZ && spacingZ.val() != 0) spacingZ.val(0);
                    if (!downSampelOrig) {
                        downSample.val(1);
                        downSampelOrig = true;
                    }
                }

                return true;
            }
        });

        spacingXY.setEnabled(vectorField.is());
        mag.setEnabled(vectorField.is());
        if (hasZ) spacingZ.setEnabled(vectorField.is());
        else overlay.setEnabled(vectorField.is());

        gd.showDialog();

        if (gd.wasCanceled()) throw new UserCanceled();

        ui = new UserInput(
                new NeighborhoodDim((int) xyR.val(), hasZ ? (int) zR.val() : 1, hasZ ? (int) layerDist.val() : 1),
                heatmap.is(),
                vectorField.is(),
                coherence.is(),
                vectorField.is() ? (int) spacingXY.val() : 0,
                vectorField.is() ? (int) mag.val() : 0,
                vectorField.is() && !hasZ ? overlay.is() : false,
                defaultTolerance,
                (int) downSample.val(),
                hasZ && vectorField.is() ? (int) spacingZ.val() : 0
        );
    }

    private boolean downSampelOrig = true;

    /**
     * Adds the help button and ties it to the generic dialog.
     *
     * @param gd The generic dialog the help frame should be added to.
     * @return The help frame.
     */
    private HelpDialog addHelpButton(GenericDialog gd) {

        HelpDialog help = new HelpDialog(gd, "Help for Neighborhood PIG");

        gd.addButton("Help", new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                help.setVisible(!help.isVisible());
                if (help.isVisible()) help.toFront();
            }
        });

        gd.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                help.dispose();
            }

        });

        return help;
    }

    /**
     * Gets the user's input.
     *
     * @return The user's input.
     */
    public UserInput getUserInput() {
        return ui;
    }

}
