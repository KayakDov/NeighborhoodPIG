package fijiPlugin;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DArray1d;
import JCudaWrapper.array.DStrideArray3d;
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
public class NeighborhoodProductSums extends Dimensions implements AutoCloseable {

//    private final Vector halfNOnes;
    private final DStrideArray3d workSpace1, workSpace2;
    private final int nRad;
    private Kernel nSum;
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
    public NeighborhoodProductSums(Handle handle, int nRad, DStrideArray3d dim) {
        super(handle, dim);

        Z = new Mapper(depth, 2, height * width * batchSize) {
            @Override
            protected int srcStride(DStrideArray3d src) {
                return src.ld() * src.linesPerLayer();
            }

            @Override
            protected int dstStride(DStrideArray3d src, DArray dst) {
                return dst.entriesPerLine() == 1 ? src.entriesPerLine() * src.linesPerLayer() * dst.ld() : dst.ld() * dst.linesPerLayer();
            }
        };

        Y = new Mapper(height, 1, depth * width * batchSize) {
            @Override
            protected int srcStride(DStrideArray3d src) {
                return 1;

            }

            @Override
            protected int dstStride(DStrideArray3d src, DArray dst) {
                return dst.entriesPerLine() == 1 ? dst.ld() : 1;
            }
        };

        X = new Mapper(width, 0, depth * height * batchSize) {
            @Override
            protected int srcStride(DStrideArray3d src) {
                return src.ld();
            }

            @Override
            protected int dstStride(DStrideArray3d src, DArray dst) {
                return dst.entriesPerLine() == 1 ? src.entriesPerLine() * dst.ld() : dst.ld();
            }
        };

        this.nRad = nRad;

        workSpace2 = dim.copyDim();
        workSpace1 = dim.copyDim();

        nSum = new Kernel("neighborhoodSum3d");
    }

    /**
     * A class to manage data for computing neighborhood sums in a specific
     * dimension.
     */
    private abstract class Mapper {

        public final int numSteps, numThreads, dirOrd;

        /**
         *
         * @param numSteps The number of steps to take in the requested
         * dimension.
         * @param srcInc The size of each step in the direction.
         * @param dirOrd The direction. x = 0, y = 1, z = 2.
         * @param numThreads The number pixels on a face perpendicular to the
         * direction.
         */
        public Mapper(int numSteps, int dirOrd, int numThreads) {
            this.numSteps = numSteps;
            this.numThreads = numThreads;
            this.dirOrd = dirOrd;
        }

        /**
         * The stride size for the source.
         *
         * @param src The data being strode through.
         * @return The stride size.
         */
        protected abstract int srcStride(DStrideArray3d src);

        /**
         * The stride size for the source.
         *
         * dst src The data being strode through.
         * @return The stride size.
         */
        protected abstract int dstStride(DStrideArray3d src, DArray dst);

        /**
         * Maps the neighborhood sums in the given dimension.
         *
         * @param n The number of threads.
         * @param src The source matrix.
         * @param dst The destination matrix.
         * @param dir The dimension, 0 for X, 1 for Y, and 2 for Z.
         * @param ldTo The increment of the the destination matrices.
         */
        public void neighborhoodSum(DStrideArray3d src, DArray dst) {


            nSum.run(handle, //TODO: make sure nSum takes in ld.
                    numThreads,
                    src,
                    P.to(dst),
                    P.to(height),
                    P.to(width),
                    P.to(depth),
                    P.to(src.ld()),
                    P.to(dst.ld()),
                    P.to(srcStride(src)),
                    P.to(dstStride(src, dst)),
                    P.to(numSteps),
                    P.to(nRad),
                    P.to(dirOrd),
                    P.to(dst.entriesPerLine() != 1)
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
    public void set(DStrideArray3d a, DStrideArray3d b, DArray1d result) {

        Kernel.run("addEBEProduct", handle, a.size(), result, //TODO: fix kernel so that target does not need to be same shape as source.
                P.to(result.ld()),
                P.to(1),
                P.to(a.entriesPerLine()),
                P.to(1),
                P.to(a),
                P.to(a.ld()),
                P.to(b),
                P.to(b.ld()),
                P.to(0)
        );

        X.neighborhoodSum(workSpace1, workSpace2);

        if (depth > 1) {
            Y.neighborhoodSum(workSpace2, workSpace1);
            Z.neighborhoodSum(workSpace1, result);
        } else
            Y.neighborhoodSum(workSpace2, result);

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
