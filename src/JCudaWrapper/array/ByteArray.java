package JCudaWrapper.array;

import static JCudaWrapper.array.IArray.empty;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

/**
 *
 * @author E. Dov Neimand
 */
public class ByteArray extends Array {

    protected ByteArray(CUdeviceptr p, int length) {
        super(p, length, PrimitiveType.BYTE);
    }

    /**
     * {@inheritDoc}
     *
     * @param handle The handle.
     * @return A copy of this array.
     */
    @Override
    public ByteArray copy(Handle handle) {
        ByteArray copy = empty(length);
        copy.set(handle, pointer, length);
        return copy;
    }

    /**
     * @see DArray#empty(int)
     * @param length The length of the array.
     * @return An empty asrray.
     */
    public static ByteArray empty(int length) {
        return new ByteArray(empty(length, PrimitiveType.BYTE), length);
    }
    
    public byte[] get(Handle handle){
        byte[] cpuArray = new byte[length];
        get(handle, Pointer.to(cpuArray), 0, 0, length);
        return cpuArray;
    }

}
