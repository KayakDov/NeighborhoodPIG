package FijiInput;

import java.awt.BorderLayout;
import java.awt.Dialog;
import java.awt.TextArea;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * A dedicated AWT Dialog for multi-line help text.
 * <p>
 * This dialog displays contextual guidance in a read-only text area
 * and can be toggled visible or hidden by client code.  It is
 * non-modal and does not block the main application.
 * </p>
 */
public class HelpDialog extends Dialog {

    /**
     * The text area used to show help messages.  Always non-editable.
     */
    private final TextArea helpTextArea;

    /**
     * The default help message to display when no specific context is set.
     */
    private static final String DEFAULT_MESSAGE =
        "Focus on a field in the main dialog for help information.";

    /**
     * Constructs a new HelpDialog with the given owner and title.
     * The dialog is modeless and uses an AWT TextArea to display help text.
     * 
     * @param owner the parent dialog over which this help dialog is centered
     * @param title the window title of the help dialog
     */
    public HelpDialog(Dialog owner, String title) {
        super(owner, title, /*modal=*/ false);
        setLayout(new BorderLayout());

        // Initialize the help text area
        helpTextArea = new TextArea(DEFAULT_MESSAGE, 8, 40,
                                    TextArea.SCROLLBARS_VERTICAL_ONLY);
        helpTextArea.setEditable(false);
        add(helpTextArea, BorderLayout.CENTER);

        // Use a fixed size so the dialog does not shrink unexpectedly
        setSize(400, 200);
        setResizable(true);
        setLocationRelativeTo(owner);

        // Hide the dialog when the user clicks the close button (X)
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                setVisible(false);
            }
        });
        
        helpTextArea.setFocusable(false);
        setFocusableWindowState(false);


    }

    /**
     * Updates the help text displayed in the dialog.  This call
     * immediately updates the read-only text area and brings the
     * dialog to the front of the window stack.
     * 
     * @param text the new help message to display
     */
    public void setHelpText(String text) {
        helpTextArea.setText(text);
        toFront();  // ensure the dialog remains visible
    }

    /**
     * Resets the help text to the default instructional message.
     */
    public void resetHelpText() {
        setHelpText(DEFAULT_MESSAGE);
    }
}
