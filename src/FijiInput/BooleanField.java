package FijiInput;

import ij.gui.GenericDialog;
import ij.gui.MultiLineLabel;
import java.awt.Checkbox; // To access boolean fields as Checkbox
import java.awt.Label;

/**
 * Represents a boolean (checkbox) input field in a GenericDialog.
 * This class abstracts the handling of checkboxes, providing methods to
 * retrieve and set their state, and control their enabled state.
 *
 * @author E. Dov Neimande
 */
public class BooleanField extends Field {
    

    /**
     * Constructs the boolean field and adds it to the GenericDialog.
     *
     * @param name The name of the field.
     * @param defaultValue The default value to be placed in the field.
     * @param gd The dialog the field is to be added to.
     * @param helpText Instructions to the user on how to use this field.
     * @param helpLabel
     */
    public BooleanField(String name, boolean defaultValue, GenericDialog gd, String helpText, HelpDialog helpLabel) {
        super(name, gd, helpText, helpLabel);
        gd.addCheckbox(name, defaultValue);
        this.awtComponent = (Checkbox) gd.getCheckboxes().get(gd.getCheckboxes().size() - 1);
        
        ((Checkbox)awtComponent).addItemListener(e -> {helpMessageFrame.setHelpText(helpText);});
    }

    /**
     * Returns the current state of the checkbox.
     *
     * @return True if the checkbox is selected, false otherwise.
     */
    public boolean is() {
        return ((Checkbox) this.awtComponent).getState();
    }
    
    /**
     * Sets the state of the checkbox.
     * @param state The boolean state to set (true for checked, false for unchecked).
     */
    public void is(boolean state) {
        ((Checkbox) this.awtComponent).setState(state);
    }
}