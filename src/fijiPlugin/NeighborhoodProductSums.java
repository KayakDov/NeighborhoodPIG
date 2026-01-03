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
     */
    public void set(P2dToF2d a, P2dToF2d b, PArray2dToD2d dst) {

        handle.runKernel("setEBEProduct",
                new PArray2dTo2d[]{workSpace1, a, b},
                dim
        );

        nSum("X", dim.height, dim.depth, workSpace1, workSpace2, nRad.xyR);
        
        if (dim.hasDepth()) {            
            nSum("Y", dim.width, dim.depth, workSpace2, workSpace1, nRad.xyR);
            nSum("Z", dim.height, dim.width, workSpace1, dst, nRad.zR.orElse(nRad.xyR));

        } else {
            nSum("Y", dim.width, dim.depth, workSpace2, dst, nRad.xyR);
        }
        
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
