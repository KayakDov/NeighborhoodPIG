package FijiInput.field;

/**
 *
 * @author E. Dov Neimand
 */
public enum VF {
    None, White, Color;
    
    public boolean is(){
        return !equals(None);
    }
}
