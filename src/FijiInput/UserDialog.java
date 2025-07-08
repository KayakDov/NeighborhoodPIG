package FijiInput;

import static FijiInput.UserInput.defaultTolerance;
import fijiPlugin.NeighborhoodDim; // Assuming NeighborhoodDim is in fijiPlugin package
import ij.ImagePlus;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import java.awt.AWTEvent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.nio.file.Path;
import java.util.Optional;

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

    private GenericDialog gd;
    private HelpDialog hf;
    private NumericField downSampleXY;
    private NumericField downSampleZ;
    private BooleanField heatmap;
    private NumericField xyR;
    private NumericField zR;
    private NumericField layerDist;
    private BooleanField coherence;
    private BooleanField vectorField;
    private BooleanField overlay;
    private NumericField spacingXY;
    private NumericField spacingZ;
    private NumericField mag;
    private DirectoryField saveToDirField;
    private boolean hasZ;

    /**
     * Creates a UserInput object from a dialog box presented to the user.
     *
     * @param imp The ImagePlus object associated with the input.
     * @throws UserCanceled If the user cancels the dialog box.
     */
    public UserDialog(ImagePlus imp) throws UserCanceled {
        hasZ = imp.getNSlices() > 1;

        gd = new GenericDialog("NeighborhoodPIG Parameters");
        hf = addHelpButton(gd);

        downSampleXY = new NumericField("Downsample factor XY:", 1, gd, 0,
                "Determines how many pixels are skipped (1 = every pixel, 2 = every other). \nIncreased values improve memory & time performance.",
                hf);

        downSampleZ = hasZ ? new NumericField("Downsample factor Z:", 1, gd, 0,
                "Determines how many pixels are skipped in the Z-direction (1 = every pixel, 2 = every other). \nIncreased values improve memory & time performance.",
                hf) : null;

        heatmap = new BooleanField("Heatmap", true, gd,
                "Generates a heatmap visualizing image orientations.",
                hf);

        xyR = new NumericField("Neighborhood xy radius", 5, gd, 0,
                "Radius (pixels) of the square neighborhood in the XY plane for structure tensor calculation.",
                hf);
        zR = hasZ ? new NumericField("Neighborhood z radius:", 5, gd, 0,
                "Radius (pixels) of the cube neighborhood in the Z-direction for 3D stacks.",
                hf) : null;
        layerDist = hasZ ? new NumericField("Z axis pixel spacing multiplier", 1, gd, 1,
                "Factor for anisotropic Z-spacing (e.g., 1.5 if Z-distance is 1.5 x XY pixel size).",
                hf) : null;

        coherence = new BooleanField("Generate coherence", false, gd,
                "Computes and displays a heatmap representing pixel coherence.",
                hf);

        vectorField = new BooleanField("Vector field", false, gd,
                "Displays vectors representing orientations.",
                hf);
        overlay = hasZ ? null : new BooleanField("Overlay Vector Field", false, gd,
                "Overlays the vector field directly on the original image.",
                hf);

        spacingXY = new NumericField("Vector Field Spacing XY:", 0, gd, 0,
                "Distance (pixels) between vectors in the xy plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf);

        spacingZ = hasZ ? new NumericField("Vector Field Spacing Z:", 0, gd, 0,
                "Distance (pixels) between vectors in the z plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf) : null;

        mag = new NumericField("Vector Field Magnitude:", 0, gd, 0,
                "Visual length (pixels) of the displayed vectors.",
                hf);

        saveToDirField = new DirectoryField("Save Directory:", "", gd,
                "Select the directory where vector data files (.dat) will be saved. Leave empty if not saving.",
                hf);

        gd.addDialogListener(dl);

        gd.showDialog();
        
        ableFields();

        if (gd.wasCanceled()) throw new UserCanceled();
        saveToDirField.savePath();

        ui = new UserInput(
                new NeighborhoodDim(
                        (int) xyR.val(),
                        (hasZ ? Optional.of((int) zR.val()) : Optional.empty()),
                        (hasZ ? Optional.of((double) layerDist.val()) : Optional.empty())
                ),
                heatmap.is(),
                vectorField.is(),
                coherence.is(),
                saveToDirField.getPath(),
                vectorField.is() && !hasZ ? Optional.of(overlay.is()) : Optional.empty(),
                vectorField.is() ? Optional.of((int) mag.val()) : Optional.empty(),
                enableSpacing() ? Optional.of((int) spacingXY.val()) : Optional.empty(),
                hasZ && enableSpacing()? Optional.of((int)spacingZ.val()) : Optional.empty(),
                (int) downSampleXY.val(),
                hasZ? Optional.of((int) downSampleZ.val()) : Optional.empty(),                
                defaultTolerance
        );
    }

    private boolean downSampleXYOrig = true;
    private boolean downSampleZOrig = true;

    /**
     * Sets fields to their initial enable/disabled setting.
     */
    private void ableFields(){
        spacingXY.setEnabled(false);
        mag.setEnabled(false);
        if (hasZ) {
            spacingZ.setEnabled(false);
            downSampleZ.setEnabled(true);
        } else {
            overlay.setEnabled(false);
        }
    }
    
    /**
     * True of xy and z spacing should be enabled. False otherwise.
     * @return True of xy and z spacing should be enabled. False otherwise.
     */
    private boolean enableSpacing() {
        return vectorField.is() || saveToDirField.getPath().isPresent();
    }

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

    /**
     * The dialog listener.
     */
    private DialogListener dl = (GenericDialog gd1, AWTEvent e) -> {
        try {

            boolean noOverlay = overlay == null || !overlay.is();

            spacingXY.setEnabled(enableSpacing());

            mag.setEnabled(vectorField.is());

            if (hasZ) {
                spacingZ.setEnabled(enableSpacing());
                downSampleZ.setEnabled(true);
            } else {
                overlay.setEnabled(vectorField.is());
                if (!vectorField.is()) overlay.is(false);
                spacingXY.setEnabled(enableSpacing());
            }

            if (vectorField.is()) {
                if (downSampleXY.val() == 1 && downSampleXYOrig) {
                    downSampleXY.val((int) xyR.val());
                    downSampleXYOrig = false;
                }
                if (hasZ && downSampleZ.val() == 1 && downSampleZOrig) {
                    downSampleZ.val((int) zR.val());
                    downSampleZOrig = false;
                }
                if (mag.val() == 0) mag.val(xyR.val());
            }
            if (enableSpacing()) {
                if (spacingXY.val() == 0) spacingXY.val(xyR.val());
                if (hasZ && spacingZ.val() == 0) spacingZ.val(xyR.val());
            }

        } catch (NumberFormatException nfe) {
        }
        return true;
    };

}
