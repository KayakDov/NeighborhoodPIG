package FijiInput.field;

import FijiInput.HelpDialog;
import ij.Prefs;
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
public abstract class Field implements FocusListener {

    protected static final String PREF_PREFIX = "NeighborhoodPIG.";

    public final String name, helpText;
    protected final GenericDialog gd;
    protected Component awtComponent;
    protected final HelpDialog helpMessageFrame;
    public final boolean isActive;

    /**
     * Constructs the field.
     *
     * @param name The name of the field.
     * @param gd The dialog the field will be placed in.
     * @param helpText Instructions to the user for how to use the field.
     * @param helpLabel The help message will be posted here when focus is on
     * this.
     * @param appearsInDialog Set to false if this will never be used. Otherwise set to
     * true.
     */
    public Field(String name, GenericDialog gd, String helpText, HelpDialog helpLabel, boolean appearsInDialog) {
        this.name = name;
        this.gd = gd;
        this.helpText = helpText;
        this.helpMessageFrame = helpLabel;
        this.isActive = appearsInDialog;
    }

    /**
     * Sets the enabled state of the AWT component associated with this field.
     *
     * @param enable true to enable the component, false to disable.
     * @return this
     */
    public Field setEnabled(boolean enable) {
        if (isActive) awtComponent.setEnabled(enable);
        return this;
    }

    /**
     * Checks if the AWT component associated with this field is enabled.
     *
     * @return true if enabled, false otherwise.
     */
    public boolean isEnabled() {
        return isActive && awtComponent.isEnabled();
    }

    /**
     * Instructions for how to use this field.
     *
     * @return Instructions for how to use this field.
     */
    public String getHelpText() {
        return helpText;
    }

    /**
     * Update the help message if focus is gained.
     *
     * @param fe
     */
    @Override
    public void focusGained(FocusEvent fe) {
        if (helpMessageFrame != null)
            helpMessageFrame.setHelpText(helpText);
    }

    /**
     * Do nothing if focus is lost.
     *
     * @param fe
     */
    @Override
    public void focusLost(FocusEvent fe) {
        if (isActive && isEnabled()) {
            saveValue();
            Prefs.savePreferences(); // Crucial: Save preferences to disk immediately
        }
    }


    /**
     * Abstract method to save the field's current value to ImageJ preferences.
     * Each concrete subclass must implement how its specific type of value
     * is saved.
     * @return this
     */
    public abstract Field saveValue();

}
