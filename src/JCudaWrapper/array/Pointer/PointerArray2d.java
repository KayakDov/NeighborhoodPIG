package JCudaWrapper.array.Pointer;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class PointerArray2d extends Array2d implements PointerArray{

    /**
     * Creates an array of pointers.
     * @param entriesPerLine The number of pointers per line.
     * @param numLines The number of lines.
     */
    public PointerArray2d(int entriesPerLine, int numLines) {
        super(entriesPerLine, numLines, Sizeof.POINTER);
    }

    /**
     * Sets this array to have the pointers proffered.
     * @param handle
     * @param srcCPUArrayOfArrays The arrays to be written here.
     * @return this.
     */
    public PointerArray2d set(Handle handle, Array... srcCPUArrayOfArrays) {
        Pointer[] cpuArrayOfPointers = Arrays.stream(srcCPUArrayOfArrays).map(array -> array.pointer()).toArray(Pointer[]::new);
        super.set(handle, Pointer.to(cpuArrayOfPointers)); 
        return this;
    }

    
    
}
