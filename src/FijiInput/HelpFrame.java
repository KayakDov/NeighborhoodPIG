package FijiInput;

import ij.gui.GenericDialog;
import ij.gui.MultiLineLabel;
import java.awt.AWTEvent;
import java.awt.Frame;
import java.awt.Panel;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import javax.swing.JFrame;

/**
 * A dedicated window for displaying multi-line help text. This Frame will
 * contain a MultiLineLabel whose text can be updated dynamically.
 */
public class HelpFrame extends JFrame {

    private final MultiLineLabel helpTextLabel;
    private static final String DEFAULT_MESSAGE = "Focus on a field in the main dialog for help information.";

    /**
     * Constructs the help frame.
     *
     * @param title The title of the frame.
     */
    public HelpFrame(String title) {
        super(title);
        setLayout(new BorderLayout()); // Use BorderLayout for simple layout
        helpTextLabel = new MultiLineLabel(DEFAULT_MESSAGE);
        Panel contentPanel = new Panel(new BorderLayout());
        contentPanel.add(helpTextLabel, BorderLayout.CENTER);
        add(contentPanel, BorderLayout.CENTER);
        setLocationRelativeTo(null);
        pack();
        setResizable(true);
        setMinimumSize(new Dimension(300, 150));
        setFocusableWindowState(true);
        setVisible(false);  // ensure initially hidden

        setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);

    }
    /**
     * Sets the help text displayed in the MultiLineLabel.
     *
     * @param text The text to display.
     */
    public void setHelpText(String text) {
        if (helpTextLabel != null) {
            helpTextLabel.setText(text);
            pack();
        }
    }

    /**
     * Resets the help text to the default message.
     */
    public void resetHelpText() {
        setHelpText(DEFAULT_MESSAGE);
    }
}
