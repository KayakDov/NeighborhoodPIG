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

    // This loads the preference when the class is initialized.
    private static String lastDirectory = Prefs.get("NeighborhoodPIG.lastDirectory", null);

    public DirectoryField(String name, String defaultPath, GenericDialog gd, String helpText, HelpDialog helpLabel) {
        super(name, gd, helpText, helpLabel);

        // Use the last directory if defaultPath is null or empty
        if ((defaultPath == null || defaultPath.trim().isEmpty()) && lastDirectory != null)
            defaultPath = lastDirectory;

        gd.addDirectoryField(name, defaultPath, 15);

        awtComponent = (TextField) gd.getStringFields().lastElement();
        
        awtComponent.addFocusListener(this);
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
     * Saves the path used this time, if one was used, for next time. This
     * method is intended to be called explicitly (e.g., when the dialog is
     * accepted).
     */
    public void savePath() {
        if (getPath().isPresent()) {
            String currentPath = getPath().get().toString();
            lastDirectory = currentPath; // Update static field
            Prefs.set("NeighborhoodPIG.lastDirectory", currentPath);
        } else {
            lastDirectory = null; // Clear static field if path is empty
            Prefs.set("NeighborhoodPIG.lastDirectory", null); // Clear preference if path is empty
        }
    }

    /**
     *
     * It's crucial for saving the preference when the user types or uses the
     * browse button.
     *
     * @param fe The FocusEvent.
     */
    @Override
    public void focusLost(FocusEvent fe) {
        savePath();
    }
}
