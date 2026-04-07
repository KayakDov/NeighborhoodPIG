package fijiPlugin;

import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Int.IArray1d;
import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.kernels.ThreadCount;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;

/**
 * This class implements element-by-element multiplication (EBEM) for
 * neighborhood-based matrix operations. It computes the sum of products from
 * neighborhoods in two input matrices, storing the results in the specified
 * vector.
 *
 * The input matrices are expected to have equal dimensions and column distances
 * (colDist).
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodProductSums implements AutoCloseable {

    private final PArray2dToD2d workSpace1, workSpace2;
    private final NeighborhoodDim nRad;
    private final Handle handle;
    private final Dimensions dim;

    /**
     * Constructs a {@code NeighborhoodProductSums} instance to compute the sum
     * of element-by-element products for neighborhoods within two matrices.
     *
     * @param handle A resource handle for creating internal matrices.
     * @param nRad Neighborhood radius; the distance from the center of a
     * neighborhood to its edge.
     * @param dim Dimensions from this will be copied.
     *
     */
    public NeighborhoodProductSums(Handle handle, NeighborhoodDim nRad, Dimensions dim) {

        this.handle = handle;
        this.dim = dim;
        this.nRad = nRad;
        
        workSpace2 = dim.emptyP2dToD2d(handle);
        workSpace1 = dim.emptyP2dToD2d(handle);

    }

    
    private void nSum(String dir, int effectiveHeight, int effectiveWidth, PArray2dToD2d src, PArray2dToD2d dst, int r){
        handle.runKernel(
                "neighborhoodSum" + dir, 
                new ThreadCount(effectiveHeight, effectiveWidth, dim.batchSize), 
                new PArray2dTo2d[]{src, dst}, 
                dim, 
                P.to(r)
        );
    }

    /**
     * Computes neighborhood element-wise multiplication of matrices a and b.
     * Divided into row and column stages for better performance. Then places in
     * result the summation of all the ebe products in the neighborhood of an
     * index pair in that index pair (column major order).
     *
     * @param a The first matrix.
     * @param b The second matrix.
     * @param dst Store the result here in column major order. Note that the
     * increment of this vector is probably not one.
     * @param numPasses The number of passes that the rolling sum makes.
     */
    public void set(P2dToF2d a, P2dToF2d b, PArray2dToD2d dst, int numPasses) {

        handle.runKernel("setEBEProduct",
                new PArray2dTo2d[]{workSpace1, a, b},
                dim
        );
        
        if(dim.hasDepth()){
            PArray2dToD2d srcI = workSpace1, dstI = workSpace2;
            for(int i = 0; i < numPasses - 1; i++){
                sum3d(srcI, dstI);
                PArray2dToD2d temp = dstI;
                dstI = srcI;
                srcI = temp;
            }
            sum3d(srcI, dst);
        } else{
            for(int i = 0; i < numPasses - 1; i++)
                sum2d(workSpace1, workSpace1, workSpace2);
            sum2d(workSpace1, dst, workSpace2);
        }
//        if(numPasses > 1){
//            double scale = Math.pow(2 * nRad.xyR + 1, -2 * dim.num()); // For 2D            
//            handle.runKernel("multiplyScalar", 
//                new ThreadCount(dim.height, dim.width, dim.depth, dim.batchSize), 
//                new PArray2dTo2d[]{dst},
//                dim,
//                P.to(scale)
//                
//            );
//        }

    }
    
    /**
     * Computes neighborhood summation of all the ebe products in the 
     * neighborhood of an index pair in that index pair (column major order).
     * Uses the src as a temporary buffer.  Values there will be overwritten.
     * @param src The grid to be summed over.
     * @param dst Store the result here in column major order. Note that the
     * increment of this vector is probably not one.  May not equal src.
     */
    private void sum3d(PArray2dToD2d src, PArray2dToD2d dst) {
        nSum("X", dim.height, dim.depth, src, dst, nRad.xyR);
        nSum("Y", dim.width, dim.depth, dst, src, nRad.xyR);
        nSum("Z", dim.height, dim.width, src, dst, nRad.zR.orElse(nRad.xyR));
    }
    
    /**
     * Computes neighborhood summation of all the ebe products in the 
     * neighborhood of an index pair in that index pair (column major order).
     * dst may not be workspace2;
     * @param src The grid to be summed over.
     * @param temp a buffer, the same size as src.
     * @param dst Store the result here in column major order. Note that the
     * increment of this vector is probably not one. May equal src, but not temp.
     */
    private void sum2d(PArray2dToD2d src, PArray2dToD2d dst, PArray2dToD2d temp) {
        nSum("X", dim.height, dim.depth, src, temp, nRad.xyR);
        nSum("Y", dim.width, dim.depth, temp, dst, nRad.xyR);
    }
    

    /**
     * Cleans up allocated memory on the gpu.
     */
    @Override
    public void close() {

        workSpace1.close();
        workSpace2.close();
    }
}
