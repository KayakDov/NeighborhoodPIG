package FijiInput.field;

/**
 *
 * @author E. Dov Neimand
 */
public enum VF {
    None, Basic, Color;
    
    public boolean is(){
        return !equals(None);
    }
}
