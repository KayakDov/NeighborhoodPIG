package FijiInput;


/**
 * An exception to be thrown when there is no image.
 * @author E. Dov Neimand
 */
public class NoImage extends Exception{

    public NoImage() {
        super("No image is open.");
    }
    
}
