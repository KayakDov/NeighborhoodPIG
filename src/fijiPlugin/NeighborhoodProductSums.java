package fijiPlugin;

import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Int.IArray1d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
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
    
    private final Mapper X, Y, Z;
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

        Z = dim.hasDepth()
                ? (new Mapper(dim.layerSize() * dim.batchSize, dim.depth, 2, nRad.zR.get()) {
            @Override
            public void nSum(Handle hand, int numThreads, PArray2dTo2d[] arrays, IArray1d dim, Pointer numSteps, Pointer nRad) {
                new Kernel("neighborhoodSumZ")
                        .run(handle, numThreads, arrays, dim, numSteps, nRad);
            }

        })
                : null;

        Y = new Mapper(dim.depth * dim.width * dim.batchSize, dim.height, 1, nRad.xyR){
            @Override
            public void nSum(Handle hand, int numThreads, PArray2dTo2d[] arrays, IArray1d dim, Pointer numSteps, Pointer nRad) {
                new Kernel("neighborhoodSumY")
                        .run(handle, numThreads, arrays, dim, numSteps, nRad);
            }

        };

        X = new Mapper(dim.depth * dim.height * dim.batchSize, dim.width, 0, nRad.xyR){
            @Override
            public void nSum(Handle hand, int numThreads, PArray2dTo2d[] arrays, IArray1d dim, Pointer numSteps, Pointer nRad) {
                new Kernel("neighborhoodSumX").run(handle, numThreads, arrays, dim, numSteps, nRad);                
            }

        };

        workSpace2 = dim.emptyP2dToD2d(handle);
        workSpace1 = dim.emptyP2dToD2d(handle);

    }

    /**
     * A class to manage data for computing neighborhood sums in a specific
     * dimension.
     */
    private abstract class Mapper {

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

            nSum(handle,
                    numThreads,
                    new PArray2dTo2d[]{src, dst},
                    dim.getGpuDim(),
                    P.to(numSteps),
                    P.to(nRad)
            );

        }

        /**
         * The kernel that summs up the elements in the neighborhood in the
         * given dimension.
         *
         * @param hand
         * @param numThreads The numvber of threads.
         * @param arrays An array containing he src and dst.
         * @param dim The dimesnisons of the data.
         * @param numSteps The numver of elements the rolling sums will be taken
         * over.
         * @param nRad The radius of each neighborhood.
         */
        public abstract void nSum(Handle hand, int numThreads, PArray2dTo2d[] arrays, IArray1d dim, Pointer numSteps, Pointer nRad);

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

        Kernel.run("setEBEProduct", handle,
                dim.size(),
                new PArray2dTo2d[]{workSpace1, a, b},
                dim
        );

        X.neighborhoodSum(workSpace1, workSpace2);

        if (dim.hasDepth()) {
            Y.neighborhoodSum(workSpace2, workSpace1);

            Z.neighborhoodSum(workSpace1, dst);

        } else {
            Y.neighborhoodSum(workSpace2, dst);
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
