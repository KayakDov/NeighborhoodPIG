package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3StrideDim;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
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
public class NeighborhoodProductSums extends TensorOrd3StrideDim implements AutoCloseable {

//    private final Vector halfNOnes;
    private final TensorOrd3Stride workSpace1, workSpace2;
    private final int nRad;
    private Kernel nSum;
    private final Dir X, Y, Z;
    
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
    public NeighborhoodProductSums(Handle handle, int nRad, TensorOrd3Stride dim) {
        super(dim);
        Z = new Dir(depth, layerDist, 2, height * width * batchSize);
        Y = new Dir(height, 1, 1, depth * width * batchSize);
        X = new Dir(width, colDist, 0, depth * height * batchSize);
        
        this.nRad = nRad;

        workSpace2 = dim.emptyCopyDimensions();
        workSpace1 = dim.emptyCopyDimensions();

        nSum = new Kernel("neighborhoodSum3d");
    }

    /**
     * A class to manage data for computing neighborhood sums in a specifif
     * dimension.
     */
    private class Dir {

        public final int numSteps, stepSize, dir, numThreads;

        public Dir(int numSteps, int stepSize, int dir, int numThreads) {
            this.numSteps = numSteps;
            this.stepSize = stepSize;
            this.dir = dir;
            this.numThreads = numThreads;
        }

        /**
         * Maps the neighborhood sums in the given dimension.
         *
         * @param n The number of threads.
         * @param from The source matrix.
         * @param to The destination matrix.
         * @param dir The dimension, 0 for X, 1 for Y, and 2 for Z.
         * @param toInc The increment of the the destination matrices.
         */
        public void mapNeighborhoodSum(DArray from, DArray to, int toInc) {

            nSum.map(handle,
                    numThreads,
                    from,
                    P.to(to),
                    P.to(height),
                    P.to(width),
                    P.to(depth),
                    P.to(stepSize),
                    P.to(numSteps),
                    P.to(toInc),
                    P.to(Math.min(nRad, numSteps)),
                    P.to(dir)
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
     * @param result Store the result here in column major order. Note that the
     * increment of this vector is probably not one.
     */
    public void set(TensorOrd3Stride a, TensorOrd3Stride b, Vector result) {

        new Vector(handle, workSpace1.dArray(), 1)
                .ebeSetProduct(
                        new Vector(handle, a.dArray(), 1),
                        new Vector(handle, b.dArray(), 1)
                );        

        X.mapNeighborhoodSum(workSpace1.dArray(), workSpace2.dArray(), 1);

        if (depth > 1) {
            Y.mapNeighborhoodSum(workSpace2.dArray(), workSpace1.dArray(), 1);
            Z.mapNeighborhoodSum(workSpace1.dArray(), result.dArray(), result.inc());
        } else
            Y.mapNeighborhoodSum(workSpace2.dArray(), result.dArray(), result.inc());

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
