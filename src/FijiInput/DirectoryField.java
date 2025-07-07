package FijiInput;

import ij.Prefs; // Import for preferences (to remember last directory)
import ij.gui.GenericDialog;
import ij.IJ;     // <--- NEW: Import ImageJ's main class to get the instance frame
import java.awt.Button;
import java.awt.FileDialog;
import java.awt.FlowLayout; // Needed for Panel layout
import java.awt.Frame;     // Needed for FileDialog constructor
import java.awt.Panel;     // Needed to group TextField and Button
import java.awt.TextField;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;       // For File operations (checking if path is directory)
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional; // To handle no directory selected

/**
 * Represents a directory selection input field in a GenericDialog.
 * This class provides a text field for the directory path and a "Browse..." button
 * to open a system file chooser for directory selection.
 *
 * @author E. Dov Neimand
 */
public class DirectoryField extends Field {

    private TextField pathTextField;
    private final Button browseButton;
    // Static field to remember the last selected directory across plugin runs
    private static String lastDirectory = Prefs.get("NeighborhoodPIG.lastDirectory", null);

    /**
     * Constructs the directory field and adds it to the GenericDialog.
     *
     * @param name The name of the field (label for the text box).
     * @param defaultPath The default path to display in the text field (can be null or empty).
     * @param gd The dialog the field is to be added to.
     * @param helpText Instructions to the user on how to use this field.
     * @param helpLabel The HelpDialog instance to display help text.
     */
    public DirectoryField(String name, String defaultPath, GenericDialog gd, String helpText, HelpDialog helpLabel) {
        super(name, gd, helpText, helpLabel); 
        
        Panel panel = new Panel(new FlowLayout(FlowLayout.LEFT, 5, 0)); 
        gd.addMessage(name);

        String initialPath = (lastDirectory != null && !lastDirectory.isEmpty()) ? lastDirectory : defaultPath;
        if (initialPath == null) initialPath = ""; // Ensure it's not null for TextField

        pathTextField = new TextField(initialPath, 20); // Set a reasonable width for the text field
        pathTextField.addFocusListener(this); // Add focus listener for help text
        panel.add(pathTextField); // Add text field to the panel

        browseButton = new Button("Browse...");
        browseButton.addFocusListener(this); // Add focus listener for help text
        panel.add(browseButton); // Add button to the panel
        
        gd.addPanel(panel);
                
        this.browseButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
        
                Frame parentFrame = IJ.getInstance();

                if (parentFrame == null) {
                    parentFrame = new Frame(); 
                    parentFrame.setUndecorated(true); // Make it invisible
                    parentFrame.setSize(0, 0);
                    parentFrame.setVisible(true);
                }

                FileDialog fd = new FileDialog(parentFrame, "Select Save Directory", FileDialog.LOAD);
                
                System.setProperty("apple.awt.fileDialogForDirectories", "true");
                
                // Set initial directory based on current text field content or last saved preference
                String dialogInitialDir = pathTextField.getText().trim();
                if (dialogInitialDir.isEmpty() && lastDirectory != null) {
                    dialogInitialDir = lastDirectory; // If text field is empty, try last saved directory
                }

                if (!dialogInitialDir.isEmpty()) {
                    File f = new File(dialogInitialDir);
                    if (f.isDirectory()) {
                        fd.setDirectory(f.getAbsolutePath()); // Set directory if it's a valid directory
                    } else if (f.getParentFile() != null && f.getParentFile().isDirectory()) {
                        fd.setDirectory(f.getParentFile().getAbsolutePath()); // Set parent directory if it's a file path
                    }
                }

                fd.setVisible(true); // Show the dialog

                // Reset the property after the dialog is closed for macOS
                System.setProperty("apple.awt.fileDialogForDirectories", "false");

                String directory = fd.getDirectory(); // Get the selected directory
                
                // Problem 1 Fix: Update text field with selected directory (fd.getDirectory() returns the path to the selected directory)
                // On Windows/Linux with older AWT, the user needs to navigate *into* the desired directory and click "Open".
                if (directory != null) {
                    Path selectedPath = Paths.get(directory);
                    pathTextField.setText(selectedPath.toString());
                    
                    // Problem 3 Fix: Store the last selected directory in ImageJ preferences
                    lastDirectory = selectedPath.toString();
                    Prefs.set("NeighborhoodPIG.lastDirectory", lastDirectory);
                }
                
                // Dispose the temporary frame only if it was created as a fallback (i.e., not IJ.getInstance())
                if (parentFrame != IJ.getInstance()) {
                    parentFrame.dispose();
                }
            }
        });
    }

    @Override
    public void setEnabled(boolean enabled) {
        pathTextField.setEnabled(enabled);
        browseButton.setEnabled(enabled);        
    }

    /**
     * Returns the currently selected directory path as an Optional.
     * If the text field is empty or contains only whitespace, it returns Optional.empty().
     *
     * @return An Optional containing the Path if a directory is selected, or Optional.empty() if not.
     */
    public Optional<Path> getPath() {
        String pathText = pathTextField.getText().trim();
        if (pathText.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Paths.get(pathText));
    }

    /**
     * Sets the text field with a new directory path.
     * Also updates the stored last directory preference.
     *
     * @param path The path to set, or null to clear the field.
     */
    public void setPath(Path path) {
        pathTextField.setText(path != null ? path.toString() : "");
        // Update lastDirectory preference when setting path programmatically
        lastDirectory = (path != null) ? path.toString() : null;
        Prefs.set("NeighborhoodPIG.lastDirectory", lastDirectory);
    }
}