package FijiInput;

import ij.gui.GenericDialog;

/**
 * A field for the input window.
 *
 * @author E. Dov Niemand
 */
public abstract class Field {

    public final String name;
    public final GenericDialog gd;    

    /**
     * Constructs the field.
     *
     * @param name The name of the field.
     * @param gd The dialog the field will be placed in.
     */
    public Field(String name, GenericDialog gd) {
        this.name = name;
        this.gd = gd;
    }

}
