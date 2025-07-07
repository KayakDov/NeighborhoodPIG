package FijiInput;

import ij.gui.GenericDialog;
import ij.gui.MultiLineLabel;
import java.awt.Label;
import java.awt.TextField; // To access numeric fields as TextField
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;

/**
 * Represents a numeric input field in a GenericDialog. This class abstracts the
 * handling of numeric fields, providing methods to retrieve and set their
 * values, and control their enabled state.
 *
 * @author E. Dov Neimand
 */
public class NumericField extends Field {

    /**
     * Constructs the numeric field and adds it to the GenericDialog.
     *
     * @param name The name of the field.
     * @param defaultValue The default value to be placed in the field.
     * @param gd The dialog the field is to be added to.
     * @param mantissa Number of digits after the decimal place.
     * @param helpText Instructions on how the user should use this field.
     * @param helpLabel Where the help text is stored.
     */
    public NumericField(String name, float defaultValue, GenericDialog gd, int mantissa, String helpText, HelpDialog helpLabel) {
        super(name, gd, helpText, helpLabel);
        gd.addNumericField(name, defaultValue, mantissa);
        awtComponent = (TextField) gd.getNumericFields().get(gd.getNumericFields().size() - 1);

        ((TextField) this.awtComponent).addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                enforceNonNegative();
            }
        });

        ((TextField) this.awtComponent).addFocusListener(this);
    }

    /**
     * {@inheritDoc }
     * @param fe 
     */
    @Override
    public void focusLost(FocusEvent fe) {
        enforceNonNegative();
    }

    
    
    /**
     * Internal method to enforce non-negative values. If the current value is
     * negative or non-numeric, it sets it to 0.
     */
    private void enforceNonNegative() {

        try {
            if (val() < 0) val(0.0f);
        } catch (NumberFormatException nfe) {

        }

    }

    /**
     * Returns the current value of the numeric field as a float.
     *
     * @return The user-selected value.
     */
    public float val() {
        return Float.parseFloat(((TextField) this.awtComponent).getText());
    }

    /**
     * Sets the value of the numeric field.
     *
     * @param value The float value to set.
     * @return this
     */
    public NumericField val(float value) {
        ((TextField) this.awtComponent).setText(String.valueOf((int) value));
        return this;
    }

}
