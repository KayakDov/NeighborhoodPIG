package JCudaWrapper.array;

import static JCudaWrapper.array.Array3d.checkNull;
import static JCudaWrapper.array.Array3d.checkPositive;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

/**
 *
 * @author E. Dov Neimand
 */
public class Array1d implements Array {

    private Pointer p;
    private final int bytesPerELement;

    public Array1d(int numElements, int bytesPerElement) {

        p = new CUdeviceptr();
        this.bytesPerELement= bytesPerElement;
        int error = JCuda.cudaMalloc(p, numElements * bytesPerElement);
    }
    
    /**
     * Copies data from a CPU array to a GPU array.
     *
     * @param to The destination GPU array.
     * @param fromCPUArray The source CPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param length The number of elements to copy.
     * @param type The type of the elements.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    protected static void copy(Handle handle, Array3d to, Pointer fromCPUArray, int toIndex, int fromIndex, int length, Array3d.PrimitiveType type) {
        checkPositive(toIndex, fromIndex, length);
        to.checkAgainstLength(toIndex + length - 1);

        int result;
   
        result = JCuda.cudaMemcpyAsync(
                to.pointer.withByteOffset(toIndex * type.size),
                fromCPUArray.withByteOffset(fromIndex * type.size),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                handle.getStream()
        );

        if (result != cudaError.cudaSuccess) {
            throw new RuntimeException("CUDA error: " + JCuda.cudaGetErrorString(result));
        }

    }
    
    
    /**
     * Copies data from this GPU array to another GPU array.
     *
     * @param to The destination GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     *
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    @Override
    public void get(Handle handle, Array3d to, int toIndex, int fromIndex, int length) {
        
        int error = JCuda.cudaMemcpyAsync(to.pointer(toIndex),
                pointer(fromIndex),
                length * bytesPerELement,
                cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                handle.getStream()
        );
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

}
