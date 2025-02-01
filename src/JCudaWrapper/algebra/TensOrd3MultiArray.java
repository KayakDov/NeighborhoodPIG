package JCudaWrapper.algebra;

import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Iterator;
import java.util.function.Supplier;

/**
 *
 * @author E. Dov Neimand
 */
public class TensOrd3MultiArray extends TensorOrd3StrideDim{
    private DPointerArray slices;

    
    
    
    public TensOrd3MultiArray(Supplier<DArray> sliceMaker, Handle handle, int height, int width, int depth, int batchSize) {
        super(handle, height, width, depth, batchSize);
        
        slices = new DPointerArray(height*width, depth*batchSize);
        
        for(int i = 0; i < batchSize * depth; i++) slices.set(handle, sliceMaker.get(), i);
    }


    @Override
    public void close() {
        Kernel.run("deepFree", handle, batchSize*depth, slices);
        slices.close();
    }

    @Override
    public Array3d array() {
        return slices;
    }
    
}
