package JCudaWrapper.array;

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
    public static Pointer to(float d) {
        return Pointer.to(new float[]{d});
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
        return Pointer.to(a.pointer());
    }
        
//    /**
//     * Points to anything with underlying data.
//     * @param d The thing with data.
//     * @return A pointer to the thing with data.
//     */
//    public static Pointer to(ColumnMajor d) {
//        return to(d.array());
//    }
        
}
