package FijiInput.field;

/**
 *
 * @author E. Dov Neimand
 */
public enum VF {
    None, Monochrome, Color;
    
    public boolean is(){
        return !equals(None);
    }
}
