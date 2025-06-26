package FijiInput;

import ij.gui.GenericDialog;
import ij.gui.MultiLineLabel;
import java.awt.AWTEvent;
import java.awt.Frame;
import java.awt.Panel;
import java.awt.BorderLayout;
import java.awt.Dialog;
import java.awt.Dimension;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;

/**
 * A dedicated window for displaying multi-line help text. This Frame will
 * contain a MultiLineLabel whose text can be updated dynamically.
 */
public class HelpDialog extends Dialog {

    private final MultiLineLabel helpText;

    private static final String DEFAULT_MESSAGE = "Focus on a field in the main dialog for help information.";

    /**
     * Constructs the help frame.
     *
     * @param title The title of the frame.
     */
    public HelpDialog(Dialog owner, String title) {
        super(owner, title, /*modal=*/ false);
        setLayout(new BorderLayout());
        helpText = new MultiLineLabel(DEFAULT_MESSAGE);
        add(helpText, BorderLayout.CENTER);
        setSize(300, 150);
        setResizable(true);
        setLocationRelativeTo(owner);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                setVisible(false);
            }
        });
    }

    /**
     * Sets the help text displayed in the MultiLineLabel.
     *
     * @param text The text to display.
     */
    public void setHelpText(String text) {
        if (helpText != null) {
            helpText.setText(text);
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
