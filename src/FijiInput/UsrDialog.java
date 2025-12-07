package FijiInput;

import FijiInput.field.NumericField;
import FijiInput.field.DirectoryField;
import FijiInput.field.BooleanField;
import static FijiInput.UsrInput.defaultTolerance;
import FijiInput.field.RadioButtonsField;
import FijiInput.field.VF;
import fijiPlugin.FijiPlugin;
import fijiPlugin.Launcher;
import fijiPlugin.NeighborhoodDim; // Assuming NeighborhoodDim is in fijiPlugin package
import ij.IJ;
import ij.ImagePlus;
import ij.Prefs;
import ij.gui.DialogListener;
import ij.gui.NonBlockingGenericDialog;
import imageWork.MyImagePlus;
import java.awt.AWTEvent;
import java.awt.Button;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Optional;

/**
 * This class is responsible for displaying a graphical user interface (GUI)
 * using ImageJ's GenericDialog to collect input parameters from the user. It
 * manages the life cycle of the dialog, user interactions, and then constructs
 * an immutable {@link UsrInput} object from the collected data.
 *
 * @author E. Dov Neimand
 */
public class UsrDialog {

    private final NonBlockingGenericDialog gd;
    private final HelpDialog hf;
    private NumericField downSampleXY;
    private NumericField downSampleZ;
    private final BooleanField heatmap;
    private NumericField xyR;
    private NumericField zR;
    private final NumericField layerDist;
    private final BooleanField coherence;
    private RadioButtonsField vectorField;
    private BooleanField overlay;
    private NumericField spacingXY;
    private NumericField spacingZ;
    private NumericField mag;
    private final DirectoryField saveToDirField;
    private final boolean hasZ;

    
    /**
     * Creates a UserInput object from a dialog box presented to the user.     
     */
    public UsrDialog() {
        this(-1, -1);
    }
    
    /**
     * Creates a UserInput object from a dialog box presented to the user.
     * @param xLoc The x location of the dialog.
     * @param yLoc The y location of the dialog.
     */
    public UsrDialog(int xLoc, int yLoc) {

        hasZ = getIJFrontImage().get().getNSlices() > 1;

        gd = new NonBlockingGenericDialog("NeighborhoodPIG Parameters");
        if(xLoc >= 0 && yLoc >=0) gd.setLocation(xLoc, yLoc);;
        
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
                "Select none for no vector field to be displayed. Color, for a colored vector field. And White for a field of white vectors.",
                hf);

        saveToDirField = new DirectoryField("Save Directory:", "", gd,
                "Select the directory where vector data files (.dat) will be saved. Leave empty if not saving.",
                hf);

        overlay = new BooleanField("Overlay Vector Field", false, gd,
                "Overlays the vector field directly on the original image.",
                hf, !hasZ).setEnabled(!hasZ && ((VF) vectorField.val().get()).is());

        spacingXY = new NumericField("Vector Field Spacing XY:", 0, gd, 0,
                "Distance (pixels) between vectors in the xy plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf).setEnabled(enableSpacing());

        if (overlay.is().orElse(false)) {
            spacingXY.val(downSampleXY.valF().get());
        }

        spacingZ = new NumericField("Vector Field Spacing Z:", 0, gd, 0,
                "Distance (pixels) between vectors in the z plane. \nAdjust to prevent crowding or sparse display.\nToo much spacing may cause an out of memmory crash.",
                hf, hasZ).setEnabled(enableSpacing());

        mag = new NumericField("Vector Field Magnitude:", 0, gd, 0,
                "Visual length (pixels) of the displayed vectors.",
                hf).setEnabled(((VF) vectorField.val().get()).is());

        updateFields();
        final DialogListener dl = (gd1, e) -> {
            updateFields();
            return true;
        };

        gd.addDialogListener(dl);

        gd.showDialog(); 

        if (gd.wasOKed()) {   
            
            if(getIJFrontImage().isPresent()) {
                new Thread(new Launcher(buildUsrInput(), Launcher.Save.fiji)).start();
                new UsrDialog(gd.getX(), gd.getY());
            } else {
                IJ.error("NeighborhoodPIG", "No image found.");

            }
        }
        if (hf != null) hf.dispose();
        

    

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
//        gd.addDialogListener(dl);
//
//        gd.showDialog();
//
//        spacingXY.setEnabled(overlay.isEnabled() || spacingXY.isEnabled());
//        if (gd.wasCanceled()) throw new UserCanceled();        
//        Prefs.savePreferences();
    }

    private void updateFields() {
        try {

            spacingXY.setEnabled(enableSpacing());

            boolean vfIs = ((VF) vectorField.val().get()).is();

            mag.setEnabled(vfIs);

            if (overlay.is().orElse(false)) {
                spacingXY.val(downSampleXY.valF().get());
            }

            overlay.setEnabled(vfIs);

            spacingZ.setEnabled(enableSpacing());

            if (downSampleOrig && vfIs) {

                if (downSampleXY.valF().get() == 0) {
                    downSampleXY.val(xyR.valI().get());
                }
                if (downSampleZ.valF().orElse(1f) == 0) {
                    downSampleZ.val(zR.valI().orElse(0));
                }
                if (mag.valF().get() == 0) {
                    mag.val(xyR.valI().get());
                }
                downSampleOrig = false;

            }

            if (enableSpacing() && spacingXY.valD().get() == 0) {
                spacingXY.val(xyR.valI().get());
                if (spacingZ.isEnabled()) {
                    spacingZ.val(zR.valI().orElse(0));
                }
            }

            if (enableSpacing() && mag.valD().orElse(1.0) == 0) {
                mag.val(spacingXY.valF().orElse(xyR.valF().orElse(3f)));
            }

        } catch (NumberFormatException nfe) {
        }
    }

    private UsrInput buildUsrInput() {
        return new UsrInput(
                getIJFrontImage().get(),
                new NeighborhoodDim(
                        xyR.valF().get().intValue(),
                        zR.valI(),
                        layerDist.valD()
                ),
                heatmap.is().get(),
                (VF) vectorField.val().get(),
                coherence.is().get(),
                saveToDirField.saveValue().getPath(),
                overlay.is(),
                mag.valI(),
                overlay.is().orElse(false) ? downSampleXY.valI() : spacingXY.valI(),
                spacingZ.valI(),
                downSampleXY.valI().get(),
                downSampleZ.valI(),
                defaultTolerance
        );
    }

    private boolean downSampleOrig = true;

    /**
     * Gets the image from ImageJ.
     *
     * @return The image currently open in imageJ.
     */
    public static Optional<ImagePlus> getIJFrontImage() {
        try {
            return Optional.of(new MyImagePlus(ij.WindowManager.getCurrentImage()));
        } catch (NullPointerException npe) {            
            return Optional.empty();
        }
    }

    /**
     * True of xy and z spacing should be enabled. False otherwise.
     *
     * @return True of xy and z spacing should be enabled. False otherwise.
     */
    private boolean enableSpacing() {
        return (!overlay.is().orElse(false))
                && (((Optional<VF>) vectorField.val()).orElse(VF.None).is() || saveToDirField.getPath().isPresent());
    }

    /**
     * Adds the help button and ties it to the generic dialog.
     *
     * @param gd The generic dialog the help frame should be added to.
     * @return The help frame.
     */
    private HelpDialog addHelpButton(NonBlockingGenericDialog gd) {

        HelpDialog help = new HelpDialog(gd, "Help for Neighborhood PIG");

        gd.addButton("Help", e -> {
            help.setVisible(!help.isVisible());
            if (help.isVisible()) {
                help.toFront();
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

}
