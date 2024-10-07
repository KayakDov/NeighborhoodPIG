package MathSupport;

import algebra.Matrix;
import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import resourceManagement.Handle;


/**
 * 2d rotation matrices.
 * @author E. Dov Neimand
 */
public class Rotation extends Matrix{

    /**
     * Creates the matrix that rotates a vector by the given angle.
     *
     * @param handle The handle.
     * @param theta The angle the matrix will rotate a vector by.     
     */
    public Rotation(Handle handle, double theta) {
        super(handle, new double[][]{
            {cos(theta), sin(theta)},
            {-sin(theta), cos(theta)}
        });
    }

    public static final Rotation id, r60, r120;

    static {
        Handle handleSt = new Handle();
        id = new Rotation(handleSt, 0);
        r60 = new Rotation(handleSt,  PI / 3);
        r120 = new Rotation(handleSt, 2 * PI / 3);
    }

}
