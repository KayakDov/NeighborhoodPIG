package FijiInput;

/**
 * An exception to be thrown if the user closed the dialog box.
 *
 * @author E. DOv Neimand
 */
public class UserCanceled extends Exception {

    /**
     * Constructs a UserCanceled exception with a default message.
     */
    public UserCanceled() {
        super("The dialogue box was canceled.");
    }

}