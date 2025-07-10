package FijiInput.field;

import FijiInput.HelpDialog;
import ij.Prefs;
import ij.gui.GenericDialog;

import java.awt.Checkbox;
import java.awt.CheckboxGroup;
import java.awt.Component;
import java.awt.Panel; // Panel is used internally by GenericDialog for radio button groups
import java.awt.event.FocusEvent;
import java.awt.event.ItemEvent; // Import ItemEvent
import java.util.Arrays;
import java.util.Optional;

/**
 * A field that represents a group of mutually exclusive radio buttons, where
 * the available options are defined by an enum. Internally, this uses
 * {@code GenericDialog.addRadioButtonGroup} and tracks the selection through a
 * {@link CheckboxGroup}.
 *
 * @param <T> The enum type representing the radio button options.
 *
 * Author: E. Dov Neimand
 */
public class RadioButtonsField<T extends Enum<T>> extends Field {

    /**
     * The enum class type used to map string labels to values.
     */
    private final Class<T> enumClass;

    /**
     * The AWT checkbox group created by GenericDialog, used to retrieve the
     * selected button from the panel.
     */
    private CheckboxGroup group;

    /**
     * Constructs a RadioButtonField using GenericDialog's radio group method.
     *
     * @param name The field label (used for preferences).
     * @param defaultValue The default selected enum value.
     * @param gd The dialog to add the radio group to.
     * @param helpText Help instructions shown on focus.
     * @param helpLabel The help text display component.
     * @param columns Number of columns for the layout of radio buttons.
     * @param isActive Whether this field is active and should be rendered.
     */
    public RadioButtonsField(String name, T defaultValue, GenericDialog gd,
            String helpText, HelpDialog helpLabel, int columns, boolean isActive) {
        super(name, gd, helpText, helpLabel, isActive);

        this.enumClass = defaultValue.getDeclaringClass();

        if (!isActive) return;

        String[] labels = Arrays.stream(enumClass.getEnumConstants()).map(e -> e.name()).toArray(String[]::new);

        String saved = Prefs.get(PREF_PREFIX + name, defaultValue.name());

        // Add the radio button group to the GenericDialog.
        // GenericDialog internally creates a Panel and CheckboxGroup for this.
        gd.addRadioButtonGroup(name, labels, 1, labels.length, saved);

        // Retrieve the Panel containing the radio buttons.
        // It's always the last component added by GenericDialog.addRadioButtonGroup.
        this.awtComponent = gd.getComponents()[gd.getComponentCount() - 1];

        // Iterate through the components within this Panel to get the individual Checkbox objects
        // and attach listeners, and also get the CheckboxGroup instance.
        boolean groupFound = false;
        for (Component c : ((Panel) awtComponent).getComponents()) {
            if (c instanceof Checkbox) {
                Checkbox checkbox = (Checkbox) c;
                
                // Attach the ItemListener to each checkbox.
                // This listener is crucial for updating help text and saving preferences
                // whenever a radio button selection changes.
                checkbox.addItemListener(e -> {
                    if (e.getStateChange() == ItemEvent.SELECTED) {
                        helpMessageFrame.setHelpText(helpText);
                        saveValue(); // Call saveValue when selection changes
                        Prefs.savePreferences(); // Crucial: Save preferences to disk immediately
                    }
                });

                // Get the CheckboxGroup instance from one of the checkboxes.
                // All checkboxes in the group will share the same instance.
                if (!groupFound) {
                    this.group = checkbox.getCheckboxGroup();
                    groupFound = true;
                }
            }
        }
        
        if (!groupFound) {
            throw new IllegalStateException("CheckboxGroup not found in the panel added by GenericDialog.addRadioButtonGroup().");
        }

        // Explicitly set the selection based on the 'saved' value.
        // This ensures the internal state of the CheckboxGroup matches the saved preference,
        // especially if there are any subtle timing issues with GenericDialog's initial setup.
        try {
            T initialSelection = Enum.valueOf(enumClass, saved);
            val(initialSelection); // Use the existing val(T value) method to set the state
        } catch (IllegalArgumentException e) {
            // If the saved value is invalid (e.g., enum name changed), default to defaultValue.
            System.err.println("Saved preference '" + saved + "' for field '" + name + "' is not a valid enum constant. Defaulting to " + defaultValue.name());
            val(defaultValue);
        }
    }

    /**
     * Constructs a RadioButtonField with default 1 column and active.
     *
     * @param name The field label (used for preferences).
     * @param defaultValue The default selected enum value.
     * @param gd The dialog to add the radio group to.
     * @param helpText Help instructions shown on focus.
     * @param helpLabel The help text display component.
     */
    public RadioButtonsField(String name, T defaultValue, GenericDialog gd,
            String helpText, HelpDialog helpLabel) {
        this(name, defaultValue, gd, helpText, helpLabel, 1, true);
    }

    /**
     * Gets the selected enum value from the radio group.
     *
     * @return An Optional containing the selected enum value, or empty if none
     * is selected.
     */
    public Optional<T> val() {
        if (!isActive || !isEnabled()) return Optional.empty();

        Checkbox selected = group.getSelectedCheckbox();
        if (selected == null) return Optional.empty();

        try {
            return Optional.of(Enum.valueOf(enumClass, selected.getLabel()));
        } catch (IllegalArgumentException e) {
            System.err.println("Error converting selected radio button label '" + selected.getLabel() + "' to enum type " + enumClass.getName() + ": " + e.getMessage());
            return Optional.empty();
        }
    }

    /**
     * Sets the selected radio button based on an enum value.
     *
     * @param value The enum constant to select.
     */
    public void val(T value) {
        if (!isActive || !isEnabled() || value == null) return;

        for (Component c : ((Panel) awtComponent).getComponents()) 
            if (c instanceof Checkbox) {
                Checkbox checkbox = (Checkbox) c;
                if (checkbox.getLabel().equals(value.name())) {
                    checkbox.setState(true);
                    return;
                }
            }
        
    }

    /**
     * Sets the enabled state of the radio buttons.
     *
     * @param enable True to enable; false to disable.
     * @return this
     */
    @Override
    public RadioButtonsField<T> setEnabled(boolean enable) {
        super.setEnabled(enable); 
        if (awtComponent instanceof Panel) 
            for (Component c : ((Panel) awtComponent).getComponents()) 
                if (c instanceof Checkbox) 
                    ((Checkbox) c).setEnabled(enable);
        
        return this;
    }

    /**
     * Saves the selected value to ImageJ preferences.
     */
    @Override
    protected void saveValue() {
        val().ifPresent(v -> Prefs.set(PREF_PREFIX + name, v.name()));
    }

    /**
     * Handles focus lost events. Saves preference value immediately.
     */
    @Override
    public void focusLost(FocusEvent fe) {        
        if (isActive && isEnabled()) {
            saveValue();
            Prefs.savePreferences();
        }
    }
}
