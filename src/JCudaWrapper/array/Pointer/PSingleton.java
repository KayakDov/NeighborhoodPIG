package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Double.DArray1d;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class PSingleton extends Singleton implements PArray {

    /**
     *
     * @param from The array this singleton is taken from.
     * @param index The index in the array this singleton is located at.
     */
    public PSingleton(PArray from, int index) {
        super(from, index);
    }

    /**
     * An empty singleton.
     *
     */
    public PSingleton() {
        super(Sizeof.POINTER);    
    }

//    /**
//     * The pointer in this singleton.
//     *
//     * @param handle
//     * @return
//     */
//    public Pointer get(Handle handle) {
//        Pointer[] arrayOfPointer = new Pointer[1];
//        get(handle, Pointer.to(arrayOfPointer));
//        return arrayOfPointer[0];
//    }

    
    /**
     * Gets the element in this singleton.
     * @param handle
     * @return The element in this singleton.
     */
    public abstract Array getVal(Handle handle);
    
    /**
     * Sets this array to have the pointers proffered.
     * @param handle
     * @param srcCPUArrayOfArrays The arrays to be written here.
     * @return this.
     */
    public PSingleton set(Handle handle, Array srcCPUArrayOfArrays) {
        Pointer[] cpuArrayOfPointers = new Pointer[1];
        cpuArrayOfPointers[0] = srcCPUArrayOfArrays.pointer();
        super.set(handle, Pointer.to(cpuArrayOfPointers)); 
        return this;
    }
}
