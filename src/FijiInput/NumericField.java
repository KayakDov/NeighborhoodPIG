package FijiInput;

import ij.gui.GenericDialog;
/**
 *
 * @author E. Dov Neimand
 */
public class NumericField extends Field{
    private final int index;
    
    /**
     * Constructs the field.
     * @param name The name of the field.
     * @param defaultValue The default value to be placed in the field.
     * @param gd THe dialog the field is to be added to.
     */
    public NumericField(String name, float defaultValue, GenericDialog gd) {
        super(name, gd);
        index = gd.getNumericFields() == null? 0: gd.getNumericFields().size();
        gd.addNumericField(name, defaultValue);
        
    }

    /**
     * The user selected value.
     * @return The user selected value.
     */
    public float getVal() {
        return Float.parseFloat(((java.awt.TextField) gd.getNumericFields().get(index)).getText());
    }
}
