package FijiInput;

import static FijiInput.UserInput.defaultTolerance;
import fijiPlugin.NeighborhoodDim; // Assuming NeighborhoodDim is in fijiPlugin package
import ij.ImagePlus;
import ij.gui.GenericDialog;
import java.awt.AWTEvent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.nio.file.Path;

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

        NumericField downSampleXY = new NumericField("Downsample factor XY:", 1, gd, 0,
                "Determines how many pixels are skipped (1 = every pixel, 2 = every other). \nIncreased values improve memory & time performance.",
                hf);

        NumericField downSampleZ = hasZ ? new NumericField("Downsample factor Z:", 1, gd, 0,
                "Determines how many pixels are skipped in the Z-direction (1 = every pixel, 2 = every other). \nIncreased values improve memory & time performance.",
                hf) : null;

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

        DirectoryField saveToDirField = new DirectoryField("Save Directory:", null, gd,
                "Select the directory where vector data files (.dat) will be saved. Leave empty if not saving.",
                hf);
                
        NumericField spacingXY = new NumericField("Vector Field Spacing XY:", 0, gd, 0,
                "Distance (pixels) between vectors in the xy plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf);

        NumericField spacingZ = hasZ ? new NumericField("Vector Field Spacing Z:", 0, gd, 0,
                "Distance (pixels) between vectors in the z plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf) : null;

        NumericField mag = new NumericField("Vector Field Magnitude:", 0, gd, 0,
                "Visual length (pixels) of the displayed vectors.",
                hf);
        
        BooleanField saveVectorsToFile = new BooleanField("Save Vectors to File", false, gd,
                "Saves the computed vectors (x y z nx ny nz) to a tab-separated text file.",
                hf);

        gd.addDialogListener((GenericDialog gd1, AWTEvent e) -> {
            try {
                spacingXY.setEnabled(vectorField.is());

                mag.setEnabled(vectorField.is());

                if (hasZ) {
                    spacingZ.setEnabled(vectorField.is());
                    downSampleZ.setEnabled(true); // Always enabled if hasZ
                } else {
                    overlay.setEnabled(vectorField.is());

                    if (vectorField.is() && overlay.is()) {
                        spacingXY.val(downSampleXY.val()).setEnabled(false);
                    } else {
                        spacingXY.setEnabled(true); // Re-enable if overlay is off
                    }

                    if (!vectorField.is()) {
                        overlay.is(false);
                    }
                }

                // Enable/disable saveVectorsToFile based on whether vectorField is enabled
                saveVectorsToFile.setEnabled(vectorField.is());

                if (vectorField.is()) {
                    if (downSampleXY.val() == 1 && downSampleXYOrig) {
                        downSampleXY.val((int) xyR.val());
                        downSampleXYOrig = false;
                    }
                    if (hasZ && downSampleZ.val() == 1 && downSampleZOrig) {
                        downSampleZ.val((int) zR.val());
                        downSampleZOrig = false;
                    }
                    if (spacingXY.val() == 0) spacingXY.val(xyR.val());
                    if (hasZ && spacingZ.val() == 0) spacingZ.val(xyR.val()); // Using xyR.val() as a placeholder for z spacing
                    if (mag.val() == 0) mag.val(xyR.val());
                } else {
                    if (spacingXY.val() != 0) spacingXY.val(0);
                    if (mag.val() != 0) mag.val(0);
                    if (hasZ && spacingZ.val() != 0) spacingZ.val(0);
                    if (!downSampleXYOrig) {
                        downSampleXY.val(1);
                        downSampleXYOrig = true;
                    }
                    if (hasZ && !downSampleZOrig) {
                        downSampleZ.val(1);
                        downSampleZOrig = true;
                    }
                    // If vectorField is disabled, ensure saveVectorsToFile is also deselected and disabled
                    if (!vectorField.is()) {
                        saveVectorsToFile.is(false);
                    }
                }

            } catch (NumberFormatException nfe) {                
            }
            return true;
        });

        spacingXY.setEnabled(vectorField.is());
        mag.setEnabled(vectorField.is());
        if (hasZ) {
            spacingZ.setEnabled(vectorField.is());
            downSampleZ.setEnabled(true);
        } else {
            overlay.setEnabled(vectorField.is());
        }
        saveVectorsToFile.setEnabled(vectorField.is()); // Set initial state for new field

        gd.showDialog();

        if (gd.wasCanceled()) throw new UserCanceled();
        
        ui = new UserInput(
                new NeighborhoodDim((int) xyR.val(), hasZ ? (int) zR.val() : 1, hasZ ? (int) layerDist.val() : 1),
                heatmap.is(),
                vectorField.is(),
                coherence.is(),
                saveToDirField.getPath(), 
                vectorField.is() ? (int) spacingXY.val() : 0,
                vectorField.is() ? (int) mag.val() : 0,
                vectorField.is() && !hasZ ? overlay.is() : false,
                defaultTolerance,
                (int) downSampleXY.val(),
                hasZ && vectorField.is() ? (int) spacingZ.val() : 0,
                hasZ ? (int) downSampleZ.val() : 1 
        );
    }

    private boolean downSampleXYOrig = true;
    private boolean downSampleZOrig = true;

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
