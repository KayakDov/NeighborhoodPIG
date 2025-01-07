package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
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
public class NeighborhoodProductSums implements AutoCloseable {

//    private final Vector halfNOnes;
    private final TensorOrd3Stride workSpace1, workSpace2;
    private final int nRad, height, width, depth, batchSize;
    private Handle hand;
    private Kernel nSum;

    /**
     * Constructs a {@code NeighborhoodProductSums} instance to compute the sum
     * of element-by-element products for neighborhoods within two matrices.
     *
     * @param handle A resource handle for creating internal matrices.
     * @param nRad Neighborhood radius; the distance from the center of a
     * neighborhood to its edge.
     * @param height The height of expected matrices. That is, matrices that
     * will be passed to the set method.
     * @param width The width of expected matrices.
     * @param depth The depth of the matrices.
     * @param batchSize The batchSize of the matrices.
     *
     */
    public NeighborhoodProductSums(Handle handle, int nRad, int height, int width, int depth, int batchSize) {
        this.nRad = nRad;
        this.height = height;
        this.batchSize = batchSize;
        this.depth = depth;
        this.width = width;
        hand = handle;

        workSpace2 = new TensorOrd3Stride(handle, height, width, depth, batchSize);
        workSpace1 = new TensorOrd3Stride(handle, height, width, depth, batchSize);

        nSum = new Kernel("neighborhoodSum3d");
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
    private void mapNeighborhoodSum(int n, DArray from, DArray to, int dir, int toInc) {
        
        int stepSize, numSteps;
        switch (dir) {
            case 0:             
                stepSize = height;
                numSteps = width;
                break;
            case 1: 
                stepSize = 1;
                numSteps = height;
                break;
            case 2: 
                stepSize = height * width;
                numSteps = depth;
                break;
            default:
                throw new RuntimeException("Direction must be 1, 2, or 3.  However, dir = " + dir);
        }
        
        nSum.map(hand,
                n,
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

        new Vector(hand, workSpace1.dArray(), 1)
                .ebeSetProduct(
                        new Vector(hand, a.dArray(), 1),
                        new Vector(hand, b.dArray(), 1)
                );

        mapNeighborhoodSum(height * depth, workSpace1.dArray(), workSpace2.dArray(), 0, 1);
        if (depth > 1) {
            mapNeighborhoodSum(depth * width, workSpace2.dArray(), workSpace1.dArray(), 1, 1);
            mapNeighborhoodSum(height * width, workSpace1.dArray(), result.dArray(), 2, result.inc());
        } else
            mapNeighborhoodSum(width, workSpace1.dArray(), result.dArray(), 1, result.inc());

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
