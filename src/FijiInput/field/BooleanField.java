package FijiInput.field;

import FijiInput.HelpDialog;
import ij.Prefs;
import ij.gui.GenericDialog;
import java.awt.Checkbox; // To access boolean fields as Checkbox
import java.util.Optional;

/**
 * Represents a boolean (checkbox) input field in a GenericDialog. This class
 * abstracts the handling of checkboxes, providing methods to retrieve and set
 * their state, and control their enabled state.
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
     * @param isActive Set to false if this will never be used. Otherwise set to
     * true.
     */
    public BooleanField(String name, boolean defaultValue, GenericDialog gd, String helpText, HelpDialog helpLabel, boolean isActive) {
        super(name, gd, helpText, helpLabel, isActive);
        if (!isActive) return;

        gd.addCheckbox(name, Prefs.get(PREF_PREFIX + name, defaultValue));
        this.awtComponent = (Checkbox) gd.getCheckboxes().get(gd.getCheckboxes().size() - 1);

        ((Checkbox) awtComponent).addItemListener(e -> {
            helpMessageFrame.setHelpText(helpText);
        });

        ((Checkbox) awtComponent).addItemListener(e -> {
            helpMessageFrame.setHelpText(helpText);
            saveValue();
            Prefs.savePreferences();
        });
    }

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
        this(name, defaultValue, gd, helpText, helpLabel, true);
    }

    /**
     * Returns the current state of the checkbox.
     *
     * @return True if the checkbox is selected, false otherwise.
     */
    public Optional<Boolean> is() {
        if (!isActive || !isEnabled()) return Optional.empty();
        return Optional.of(((Checkbox) this.awtComponent).getState());
    }

    /**
     * Sets the state of the checkbox.
     *
     * @param state The boolean state to set (true for checked, false for
     * unchecked).
     */
    public void is(boolean state) {
        if (isActive) ((Checkbox) this.awtComponent).setState(state);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public BooleanField setEnabled(boolean enable) {

        super.setEnabled(enable);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Field saveValue() {
        is().ifPresent(state -> Prefs.set(PREF_PREFIX + name, state));
        return this;
    }

}