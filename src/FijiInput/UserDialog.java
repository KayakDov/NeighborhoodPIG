package FijiInput;

import FijiInput.field.NumericField;
import FijiInput.field.DirectoryField;
import FijiInput.field.BooleanField;
import static FijiInput.UserInput.defaultTolerance;
import FijiInput.field.RadioButtonsField;
import FijiInput.field.VF;
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

    private final GenericDialog gd;
    private final HelpDialog hf;
    private NumericField downSampleXY;
    private NumericField downSampleZ;
    private BooleanField heatmap;
    private NumericField xyR;
    private NumericField zR;
    private NumericField layerDist;
    private BooleanField coherence;
    private RadioButtonsField vectorField;
    private BooleanField overlay;
    private NumericField spacingXY;
    private NumericField spacingZ;
    private NumericField mag;
    private final DirectoryField saveToDirField;
    private final boolean hasZ;

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

        downSampleZ = new NumericField("Downsample factor Z:", 1, gd, 0,
                "Determines how many pixels are skipped in the Z-direction (1 = every pixel, 2 = every other). \nIncreased values improve memory & time performance.",
                hf, hasZ);

        heatmap = new BooleanField("Heatmap", true, gd,
                "Generates a heatmap visualizing image orientations.",
                hf);

        xyR = new NumericField("Neighborhood xy radius", 5, gd, 0,
                "Radius (pixels) of the square neighborhood in the XY plane for structure tensor calculation.",
                hf);
        zR = new NumericField("Neighborhood z radius:", 5, gd, 0,
                "Radius (pixels) of the cube neighborhood in the Z-direction for 3D stacks.",
                hf, hasZ);
        layerDist = new NumericField("Z axis pixel spacing multiplier", 1, gd, 1,
                "Factor for anisotropic Z-spacing (e.g., 1.5 if Z-distance is 1.5 x XY pixel size).",
                hf, hasZ);

        coherence = new BooleanField("Generate coherence", false, gd,
                "Computes and displays a heatmap representing pixel coherence.",
                hf);

        vectorField = new RadioButtonsField("Vector Field", VF.None, gd, 
                "Select none for no vector field to be displayed. Color, for a colored vector field. And White for a field of white vectors."
                , hf);

        saveToDirField = new DirectoryField("Save Directory:", "", gd,
                "Select the directory where vector data files (.dat) will be saved. Leave empty if not saving.",
                hf);

        overlay = new BooleanField("Overlay Vector Field", false, gd,
                "Overlays the vector field directly on the original image.",
                hf, !hasZ).setEnabled(!hasZ && ((VF)vectorField.val().get()).is());

        spacingXY = new NumericField("Vector Field Spacing XY:", 0, gd, 0,
                "Distance (pixels) between vectors in the xy plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf).setEnabled(enableSpacing());

        if(overlay.is().orElse(false)) spacingXY.val(downSampleXY.valF().get());
        
        spacingZ = new NumericField("Vector Field Spacing Z:", 0, gd, 0,
                "Distance (pixels) between vectors in the z plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf, hasZ).setEnabled(enableSpacing());

        mag = new NumericField("Vector Field Magnitude:", 0, gd, 0,
                "Visual length (pixels) of the displayed vectors.",
                hf).setEnabled(((VF)vectorField.val().get()).is());

        gd.addDialogListener(dl);

        gd.showDialog();

        if (gd.wasCanceled()) throw new UserCanceled();

        ui = new UserInput(
                new NeighborhoodDim(
                        xyR.valF().get().intValue(),
                        zR.valI(),
                        layerDist.valD()
                ),
                heatmap.is().get(),
                (VF)vectorField.val().get(),
                coherence.is().get(),
                saveToDirField.getPath(),
                overlay.is(),
                mag.valI(),
                spacingXY.valI(),
                spacingZ.valI(),
                downSampleXY.valI().get(),
                downSampleZ.valI(),
                defaultTolerance
        );
    }

    private boolean downSampleOrig = true;

    /**
     * True of xy and z spacing should be enabled. False otherwise.
     *
     * @return True of xy and z spacing should be enabled. False otherwise.
     */
    private boolean enableSpacing() {
        return (!overlay.is().orElse(false)) && 
                (((Optional<VF>)vectorField.val()).orElse(VF.None).is() || saveToDirField.getPath().isPresent());
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

            spacingXY.setEnabled(enableSpacing());

            boolean vfIs = ((VF)vectorField.val().get()).is();
            
            mag.setEnabled(vfIs);
            overlay.setEnabled(vfIs);

            spacingZ.setEnabled(enableSpacing());

            

            if (downSampleOrig && vfIs) {

                if (downSampleXY.valF().get() == 0) downSampleXY.val(xyR.valI().get());
                if (downSampleZ.valF().orElse(1f) == 0) downSampleZ.val(zR.valI().orElse(0));
                if (mag.valF().get() == 0) mag.val(xyR.valI().get());
                downSampleOrig = false;

            }

            if (enableSpacing() && spacingXY.valD().get() == 0) {
                spacingXY.val(xyR.valI().get());
                spacingZ.val(zR.valI().orElse(0));
            }
            
            if(overlay.is().orElse(false)) spacingXY.val(downSampleXY.valF().get());

        } catch (NumberFormatException nfe) {
        }
        return true;
    };

}
