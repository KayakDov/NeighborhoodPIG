package FijiInput;

import ij.gui.GenericDialog;
import java.awt.Component; // Import AWT Component for setEnabled
import java.awt.Label;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;

/**
 * A field for the input window. This abstract class provides a common base for
 * different types of input fields within a GenericDialog, abstracting away the
 * direct index management and providing methods to get/set values and control
 * enablement.
 *
 * @author E. Dov Niemand
 */
public abstract class Field {

    public final String name, helpText;
    protected final GenericDialog gd;
    protected Component awtComponent;
    protected final HelpDialog helpMessageFrame;

    /**
     * Constructs the field.
     *
     * @param name The name of the field.
     * @param gd The dialog the field will be placed in.
     * @param helpText Instructions to the user for how to use the field.
     * @param helpLabel The help message will be posted here when focus is on
     * this.
     */
    public Field(String name, GenericDialog gd, String helpText, HelpDialog helpLabel) {
        this.name = name;
        this.gd = gd;
        this.helpText = helpText;
        this.helpMessageFrame = helpLabel;
    }

    /**
     * Sets the enabled state of the AWT component associated with this field.
     *
     * @param enable true to enable the component, false to disable.
     */
    public void setEnabled(boolean enable) {
        if (awtComponent != null) {
            awtComponent.setEnabled(enable);
        }
    }

    /**
     * Checks if the AWT component associated with this field is enabled.
     *
     * @return true if enabled, false otherwise.
     */
    public boolean isEnabled() {
        return awtComponent != null && awtComponent.isEnabled();
    }

    /**
     * Instructions for how to use this field.
     *
     * @return Instructions for how to use this field.
     */
    public String getHelpText() {
        return helpText;
    }

}
