package FijiInput.field;

import FijiInput.HelpDialog;
import ij.Prefs;
import ij.gui.GenericDialog;
import java.awt.TextField; // To access numeric fields as TextField
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.util.Optional;

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
     * @param isActive Set to false if this will never be used, otherwise true.
     */
    public NumericField(String name, float defaultValue, GenericDialog gd, int mantissa, String helpText, HelpDialog helpLabel, boolean isActive) {
        super(name, gd, helpText, helpLabel, isActive);
        if (!isActive) return;
        gd.addNumericField(name, Prefs.get(PREF_PREFIX + name, defaultValue), mantissa);
        awtComponent = (TextField) gd.getNumericFields().get(gd.getNumericFields().size() - 1);

        ((TextField) this.awtComponent).addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                enforceNonNegative();
                saveValue();
                Prefs.savePreferences();
            }
        });

        ((TextField) this.awtComponent).addFocusListener(this);
    }

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
        this(name, defaultValue, gd, mantissa, helpText, helpLabel, true);
    }

    /**
     * {@inheritDoc }
     *
     * @param fe
     */
    @Override
    public void focusLost(FocusEvent fe) {
        enforceNonNegative();
        super.focusLost(fe);
    }

    /**
     * Internal method to enforce non-negative values. If the current value is
     * negative or non-numeric, it sets it to 0.
     */
    private void enforceNonNegative() {

        try {
            if (isActive && valF().get() < 0) val(0.0f);
        } catch (NumberFormatException nfe) {
            if (isActive) val(0f);
        }

    }

    /**
     * Returns the current value of the numeric field as a float.
     *
     * @return The user-selected value.
     */
    public Optional<Float> valF() {
        if (!isActive) return Optional.empty();
        return Optional.of(Float.valueOf(((TextField) this.awtComponent).getText()));
    }

    /**
     * Returns the current value of the numeric field as a float.
     *
     * @return The user-selected value.
     */
    public Optional<Double> valD() {
        if (!isActive) return Optional.empty();
        return Optional.of(Double.valueOf(((TextField) this.awtComponent).getText()));
    }

    /**
     * Returns the current value of the numeric field as a float.
     *
     * @return The user-selected value.
     */
    public Optional<Integer> valI() {
        if (!isActive) return Optional.empty();
        try {
            return Optional.of(Integer.valueOf(((TextField) this.awtComponent).getText()));
        } catch (NumberFormatException nfe) {
            throw new NumberFormatException("It looks like you put a non integer value in '" + name + "'.  This should be an integer.  Here is the help text: " + helpText);
        }
    }

    /**
     * Sets the value of the numeric field.
     *
     * @param value The float value to set.
     * @return this
     */
    public NumericField val(float value) {
        if (!isActive) return this;
        ((TextField) this.awtComponent).setText(String.valueOf((int) value));
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public NumericField setEnabled(boolean enable) {
        super.setEnabled(enable);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    protected void saveValue() {
        valF().ifPresent(value -> Prefs.set(PREF_PREFIX + name, value));
    }
}
