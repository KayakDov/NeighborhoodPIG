package FijiInput;

import ij.Prefs;
import ij.gui.GenericDialog;
import java.awt.TextField;
import java.awt.event.FocusEvent;
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

    /**
     * Constructs the directory field and adds it to the GenericDialog.
     *
     * @param name The name of the field (label for the text box).
     * @param defaultPath The default path to display in the text field (can be
     * null or empty).
     * @param gd The dialog the field is to be added to.
     * @param helpText Instructions to the user on how to use this field.
     * @param helpLabel The HelpDialog instance to display help text.
     */
    public DirectoryField(String name, String defaultPath, GenericDialog gd, String helpText, HelpDialog helpLabel) {
        super(name, gd, helpText, helpLabel);

        gd.addDirectoryField(name, defaultPath, 15);

        this.awtComponent = (TextField)gd.getStringFields().lastElement();

        this.awtComponent.addFocusListener(this); // Add focus listener for help text and dialog updates
    }
    /**
     * Returns the currently selected directory path as an Optional. If the text
     * field is empty or contains only whitespace, it returns Optional.empty().
     *
     * @return An Optional containing the Path if a directory is selected, or
     * Optional.empty() if not.
     */
    public Optional<Path> getPath() {
        // Cast awtComponent to TextField to access getText()
        String pathText = ((TextField) this.awtComponent).getText().trim();
        if (pathText.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Paths.get(pathText));
    }
    
    /**
     * Saves the path used this time, if one was used, for next time.
     */
    public void savePath(){
        if(getPath().isPresent()) Prefs.set("NeighborhoodPIG.lastDirectory", getPath().get().toString());

    }

}
