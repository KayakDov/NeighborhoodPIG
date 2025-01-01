package JCudaWrapper.array;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.algebra.VectorsStride;
import jcuda.Pointer;

/**
 * Quick access to cpu pointers.
 * @author E. DOv Neimand
 */
public class P {

    /**
     * A pointer to a singleton array containing d.
     *
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    public static Pointer to(int d) {
        return Pointer.to(new int[]{d});
    }
    
    /**
     * A pointer to a singleton array containing d.
     *
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    public static Pointer to(double d) {
        return Pointer.to(new double[]{d});
    }
    
    /**
     * A pointer to a singleton array containing d.
     *
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    public static Pointer to(boolean d) {
        return Pointer.to(new int[]{d?1:0});
    }
    
    /**
     * A pointer to a singleton array containing d.
     *
     * @param <T> extends an array.
     * @param a A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    public static <T extends Array> Pointer to(T a) {
        return Pointer.to(a.pointer);
    }
    
    /**
     * Pointer to the data underlying a vector.
     * @param d
     * @return 
     */
    public static Pointer to(Vector d) {
        return to(d.dArray());
    }
    
    /**
     * Pointer to the data underlying a vectorsStride.
     * @param d
     * @return 
     */
    public static Pointer to(VectorsStride d) {
        return to(d.dArray());
    }
    
    public static Pointer to(MatricesStride d) {
        return to(d.dArray());
    }
    
    public static Pointer to(TensorOrd3Stride d) {
        return to(d.dArray());
    }
    public static Pointer to(Matrix d) {
        return to(d.dArray());
    }
    
}
