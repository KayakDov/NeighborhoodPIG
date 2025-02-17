package fijiPlugin;

import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DStrideArray3d;
import JCudaWrapper.array.IArray1d;
import java.util.Arrays;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient extends Dimensions implements AutoCloseable {


    public final DStrideArray3d x, y, z;

    /**
     * Compute gradients of an image in both the x and y directions. Gradients
     * are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param handle The context
     * @param pic The pixel intensity values matrix.
     
     *
     */
    public Gradient(Handle handle, DStrideArray3d pic) {
        super(handle, pic);

        x = pic.copyDim();
        y = pic.copyDim();
        z = pic.copyDim();

        //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
        int[] dimensions = new int[]{
            height,   //0
            width,    //1
            depth,    //2
            batchSize,//3 
            height*width,//4
            tensorSize(),//5 
            tensorSize() * batchSize,//6
            pic.ld() //7
        };               
        
        try (IArray1d dim = new IArray1d(8).set(handle, dimensions)) {
            
            Kernel.run("batchGradients", handle,
                    x.size()*3,
                    pic,
                    P.to(dim),
                    P.to(x), P.to(x.ld()),
                    P.to(y), P.to(y.ld()),
                    P.to(z), P.to(z.ld())
            );
        }        
    }

   /**
    * {@inheritDoc }
    */
    @Override
    public void close() {
        x.close();
        y.close();
        z.close();
    }

    /**
     * The number of pixels for which the gradient is calculated.
     *
     * @return The number of pixels for which the gradient is calculated.
     */
    public int size() {
        return height * width * depth * batchSize;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        return super.toString() + "\nd\\dx =\n" + x.toString() + "\nd\\dy = \n" + y.toString() + "\nd\\dz = \n" + z.toString();
    }
    

}
