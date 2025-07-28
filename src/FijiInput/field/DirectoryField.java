package FijiInput.field;

import FijiInput.HelpDialog;
import ij.Prefs;
import ij.gui.GenericDialog;
import java.awt.Checkbox;
import java.awt.TextField;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

/**
 * Represents a directory selection input field in a GenericDialog. This class
 * provides a text field for the directory path and a "Browse..." button to open
 * a system file chooser for directory selection.
 *
 * @author E. Dov Neimand
 */
public class DirectoryField extends Field {

    public DirectoryField(String name, String defaultPath, GenericDialog gd, String helpText, HelpDialog helpLabel) {
        super(name, gd, helpText, helpLabel, true);

        gd.addDirectoryField(name, Prefs.get(PREF_PREFIX + name, defaultPath), 15);

        awtComponent = (TextField) gd.getStringFields().lastElement();

        awtComponent.addFocusListener(this);
        
        ((TextField)awtComponent).addActionListener(e -> {
            helpMessageFrame.setHelpText(helpText);
            saveValue();
            Prefs.savePreferences();
        });
    }

    /**
     * Returns the currently selected directory path as an Optional. If the text
     * field is empty or contains only whitespace, it returns Optional.empty().
     *
     * @return An Optional containing the Path if a directory is selected, or
     * Optional.empty() if not.
     */
    public Optional<Path> getPath() {
        String pathText = ((TextField) this.awtComponent).getText().trim();
        if (pathText.isEmpty()) return Optional.empty();
        return Optional.of(Paths.get(pathText));
    }

    /**
     * {@inheritDoc
     */
    @Override
    public DirectoryField saveValue() {
        getPath().ifPresent(path -> Prefs.set(PREF_PREFIX + name, path.toString()));
        return this;
    }
}
