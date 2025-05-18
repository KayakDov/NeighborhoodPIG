package fijiPlugin;

import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;

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
public class NeighborhoodProductSums extends Dimensions implements AutoCloseable {

//    private final Vector halfNOnes;
    private final PArray2dToD2d workSpace1, workSpace2;
    private final Kernel nSum;
    private final Mapper X, Y, Z;

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
    public NeighborhoodProductSums(Handle handle, NeighborhoodDim nRad, PArray2dToD2d dim) {
        super(handle, dim);

        Z = new Mapper(height * width * batchSize, depth, 2, nRad.zR);

        Y = new Mapper(depth * width * batchSize, height, 1, nRad.xyR);

        X = new Mapper(depth * height * batchSize, width, 0, nRad.xyR);

        workSpace2 = dim.copyDim(handle);
        workSpace1 = dim.copyDim(handle);

        nSum = new Kernel("neighborhoodSum3d", Kernel.Type.PTR_TO_DOUBLE_2D);
    }

    /**
     * The Z stride size in a matrix.
     */
    private static int matZStride(FArray mat) {
        return mat.ld() * mat.linesPerLayer();
    }    

    /**
     * A class to manage data for computing neighborhood sums in a specific
     * dimension.
     */
    private class Mapper {

        public final int numSteps, numThreads, dirOrd, nRad;

        /**
         *
         * @param numSteps The number of steps to take in the requested
         * dimension.
         * @param srcInc The size of each step in the direction.
         * @param dirOrd The direction. x = 0, y = 1, z = 2.
         * @param numThreads The number pixels on a face perpendicular to the
         * direction.
         */
        public Mapper(int numThreads, int numSteps, int dirOrd, int nRad) {
            this.numSteps = numSteps;
            this.numThreads = numThreads;
            this.dirOrd = dirOrd;
            this.nRad = nRad;
        }

        /**
         * Maps the neighborhood sums in the given dimension.
         *
         * @param n The number of threads.
         * @param src The source matrix.
         * @param dst The destination matrix.
         * @param dir The dimension, 0 for X, 1 for Y, and 2 for Z.
         * @param ldTo The increment of the the destination matrices.
         */
        public void neighborhoodSum(PArray2dToD2d src, PArray2dToD2d dst) {
            
            nSum.run(handle,
                    numThreads,
                    new PArray2dTo2d[]{src, dst},
                    NeighborhoodProductSums.this,
                    P.to(numSteps),
                    P.to(nRad),
                    P.to(dirOrd)                    
            );            
        }

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
    public void set(PArray2dToD2d a, PArray2dToD2d b, PArray2dToD2d dst) {
        
        Kernel.run("addEBEProduct", handle, 
                a.size(), 
                new PArray2dTo2d[]{workSpace1,a, b},
                this,
                P.to(1.0),                
                P.to(0.0)
        );
           
        X.neighborhoodSum(workSpace1, workSpace2);

        if (depth > 1) {
            Y.neighborhoodSum(workSpace2, workSpace1);
            Z.neighborhoodSum(workSpace1, dst);
        } else
            Y.neighborhoodSum(workSpace2, dst);
    }

    /**
     * Cleans up allocated memory on the gpu.
     */
    @Override
    public void close() {

        workSpace1.close();
        workSpace2.close();
        nSum.close();
    }
}
