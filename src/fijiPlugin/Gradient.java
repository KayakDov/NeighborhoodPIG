package fijiPlugin;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
import JCudaWrapper.resourceManagement.Handle;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient implements AutoCloseable{

    private TensorOrd3Stride dX, dY, dZ;

    /**
     * Computemutty gradients of an image in both the x and y directions.
     * Gradients are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     *
     */
    public Gradient(TensorOrd3Stride pic, Handle hand) {
        
        dX = pic.emptyCopyDimensions();
        dY = pic.emptyCopyDimensions();
        dZ = pic.emptyCopyDimensions();

        KernelManager.get("batchGradients").map(hand, 
                3*pic.dArray().length, 
                pic.dArray(),
                IArray.cpuPointer(pic.height), 
                IArray.cpuPointer(pic.width), 
                IArray.cpuPointer(pic.depth), 
                IArray.cpuPointer(pic.batchSize),
                dX.dArray().pToP(), 
                dY.dArray().pToP(), 
                dZ.dArray().pToP()
        );
        
    }
    
    public static void main(String[] args) {
        try(Handle hand = new Handle(); DArray array = new DArray(hand, 1,2,3, 4,5,5, 4,3,2, 1,3,3, 3,3,3)){
            TensorOrd3Stride tenStr = new Matrix(hand, array, 3, 5).repeating(1);
            
            try(Gradient grad = new Gradient(tenStr, hand)){
                
                System.out.println("dX: " + grad.dX.dArray().toString());
                
            }
        }
    }

    /**
     * An  x gradient matrix.
     *
     * @return An  x gradient matrix.
     */
    public TensorOrd3Stride x() {
        return dX;
    }

    /**
     * An  y gradient matrix.
     *
     * @return A y gradient matrix.
     */
    public TensorOrd3Stride y() {
        return dY;
    }
        
    /**
     * A y gradient matrix.
     *
     * @return A y gradient matrix.
     */
    public TensorOrd3Stride z() {
        return dZ;
    }

    @Override
    public void close() {
        dX.close();
        dY.close();
        dZ.close();
    }
    
}
