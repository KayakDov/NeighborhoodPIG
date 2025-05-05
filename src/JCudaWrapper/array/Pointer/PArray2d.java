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
public abstract class PArray2d extends Array2d implements PArray{

    /**
     * Creates an array of pointers.
     * @param entriesPerLine The number of pointers per line.
     * @param numLines The number of lines.
     */
    public PArray2d(int entriesPerLine, int numLines) {
        super(entriesPerLine, numLines, Sizeof.POINTER);
    }

    /**
     * Sets this array to have the pointers proffered.
     * @param handle
     * @param srcCPUArrayOfArrays The arrays to be written here.
     * @return this.
     */
    public PArray2d set(Handle handle, Array... srcCPUArrayOfArrays) {
        Pointer[] cpuArrayOfPointers = Arrays.stream(srcCPUArrayOfArrays).map(array -> array.pointer()).toArray(Pointer[]::new);
        super.set(handle, Pointer.to(cpuArrayOfPointers)); 
        return this;
    }

    
    
}
