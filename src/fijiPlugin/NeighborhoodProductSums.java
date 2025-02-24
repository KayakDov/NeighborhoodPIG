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
    public NeighborhoodProductSums(Handle handle, NeighborhoodDim nRad, DStrideArray3d dim) {
        super(handle, dim);

        Z = new Mapper(height * width * batchSize, depth, 2, nRad.z) {

            @Override
            protected int srcStride(DStrideArray3d src) {
                return matZStride(src);
            }

            @Override
            protected int dstStride(DStrideArray3d src, DArray dst) {
                return dst.is1D()
                        ? src.entriesPerLine() * src.linesPerLayer() * dst.ld()
                        : matZStride(src);
            }
        };

        Y = new Mapper(depth * width * batchSize, height, 1, nRad.xy) {
            @Override
            protected int srcStride(DStrideArray3d src) {
                return 1;

            }

            @Override
            protected int dstStride(DStrideArray3d src, DArray dst) {
                return dst.is1D() ? dst.ld() : 1;
            }
        };

        X = new Mapper(depth * height * batchSize, width, 0, nRad.xy) {
            @Override
            protected int srcStride(DStrideArray3d src) {
                return src.ld();
            }

            @Override
            protected int dstStride(DStrideArray3d src, DArray dst) {
                return (dst.is1D() ? src.entriesPerLine() : 1) * dst.ld();
            }
        };

        workSpace2 = dim.copyDim();
        workSpace1 = dim.copyDim();

        nSum = new Kernel("neighborhoodSum3d");
    }

    /**
     * The Z stride size in a matrix.
     */
    private static int matZStride(DArray mat) {
        return mat.ld() * mat.linesPerLayer();
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
         *
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
                    P.to(!dst.is1D())
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
    public void set(DStrideArray3d a, DStrideArray3d b, DArray dst) {
        //TODO: fix workspace 1 height and ld
        Kernel.run("addEBEProduct", handle, 
                a.size(), 
                workSpace1,
                P.to(workSpace1.ld()),
                P.to(workSpace1.entriesPerLine()),
                P.to(a.entriesPerLine()),
                P.to(1.0),
                P.to(a),
                P.to(a.ld()),
                P.to(b),
                P.to(b.ld()),
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
