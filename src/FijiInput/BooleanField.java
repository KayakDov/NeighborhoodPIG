package FijiInput;

import ij.gui.GenericDialog;
import java.awt.Checkbox;

/**
 *
 * @author E. Dov Neimande
 */
public class BooleanField extends Field {

    private final int index;

    /**
     * Constructs the field.
     *
     * @param name The name of the field.
     * @param defaultValue The default value to be placed in the field.
     * @param gd THe dialog the field is to be added to.
     */
    public BooleanField(String name, boolean defaultValue, GenericDialog gd) {
        super(name, gd);
        index = gd.getCheckboxes() == null? 0 : gd.getCheckboxes().size();
        gd.addCheckbox(name, defaultValue);
    }

    /**
     * The user selected value.
     *
     * @return The user selected value.
     */
    public boolean is() {
        return ((Checkbox)gd.getCheckboxes().get(index)).getState();
    }

}
