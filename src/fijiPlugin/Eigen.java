package fijiPlugin;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Int.IStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;

/**
 * A set of 3x3 matrices, their eigenvectors and values.
 *
 * @author E. Dov Neimand
 */
public class Eigen extends Dimensions implements AutoCloseable {//TODO: maybe incorporate this up into StructureTensorMatrices.

    private final double tolerance;
    private final int downsampleFactorXY;

    /**
     * Each layer in this matrix is for a different pixel, in column major
     * order.
     */
    public final PArray2dToD2d[][] mat;

    /**
     * These are organized in the same columns, layers, and grids as the initial
     * picture. The rows are changed so that each set of 3 eigenvectors are
     * consecutive in each column.
     */
    public final PArray2dToD2d values;//TODO: there may not be any need to store eigenvalues.

    /**
     * The first eigevector of each structureTensor with mathcin columns and
     * layers as the original pixels, and rows * 3.
     */
    public final PArray2dToD2d vectors;

    /**
     *
     * @param handle
     * @param dim
     * @param downSampleFactorXY 1 in every how many pixels get evaluated in the
     * x and y dimensions.
     * @param tolerance
     */
    public Eigen(Handle handle, Dimensions dim, int downSampleFactorXY, double tolerance) {
        super(dim);
        this.downsampleFactorXY = downSampleFactorXY;
        this.tolerance = tolerance;
        mat = new PArray2dToD2d[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++)
                mat[j][i] = mat[i][j] = dim.empty();

        values = new PArray2dToD2d(dim.depth, dim.batchSize, (dim.height / downSampleFactorXY) * 3, dim.width / downSampleFactorXY);
        vectors = values.copyDim();

    }

    /**
     * All the values of all the structure tensors at the given indices.
     *
     * @param row The row of the desired vector.
     * @param col The column of the desired vector.
     * @return All the values of all the structure tensors at the given indices.
     */
    public PArray2dToD2d at(int row, int col) {
        return mat[row][col];
    }


    /**
     * Sets the eigen values and the eigen vectors at the requested index.
     *
     * @param eigenInd If the index is 0, then the eigen vectors that corespond
     * to the greatest eigenvalue, if the index is one, then the 2nd greatest.
     * @param coherence Where to store the coherence data.
     * @param azimuth Location to load the azimuthal angles.
     * @param zenith Location to load the zenith angles.
     */
    public void set(int eigenInd, PArray2dToD2d coherence, PArray2dToD2d azimuth, PArray2dToD2d zenith) {
        Kernel.run("eigenBatch", handle,
                size(),
                
                mat[0][0]      , P.to(mat[0][0].targetLD()), P.to(mat[0][0].targetLD().ld()), P.to(mat[0][0].ld()),
                P.to(mat[0][1]), P.to(mat[0][1].targetLD()), P.to(mat[0][1].targetLD().ld()), P.to(mat[0][1].ld()),
                P.to(mat[0][2]), P.to(mat[0][2].targetLD()), P.to(mat[0][2].targetLD().ld()), P.to(mat[0][2].ld()),
                P.to(mat[1][1]), P.to(mat[1][1].targetLD()), P.to(mat[1][1].targetLD().ld()), P.to(mat[1][1].ld()),
                P.to(mat[1][2]), P.to(mat[1][2].targetLD()), P.to(mat[1][2].targetLD().ld()), P.to(mat[1][2].ld()),
                P.to(mat[2][2]), P.to(mat[2][2].targetLD()), P.to(mat[2][2].targetLD().ld()), P.to(mat[2][2].ld()),

                P.to(values),    P.to(values.targetLD()),    P.to(values.targetLD().ld()),    P.to(values.ld()),
                P.to(vectors),   P.to(vectors.targetLD()),   P.to(vectors.targetLD().ld()),   P.to(vectors.ld()),

                P.to(coherence), P.to(coherence.targetLD()), P.to(coherence.targetLD().ld()), P.to(coherence.ld()),
                P.to(azimuth),   P.to(azimuth.targetLD()),   P.to(azimuth.targetLD().ld()),   P.to(azimuth.ld()),
                P.to(zenith),    P.to(zenith.targetLD()),    P.to(zenith.targetLD().ld()),    P.to(zenith.ld()),
                
                P.to(height), P.to(width), P.to(depth),                
                P.to(downsampleFactorXY), P.to(eigenInd),
                P.to(tolerance)
        );
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        for (int i = 0; i < mat.length; i++)
            for (int j = i; j < mat[0].length; j++)
                mat[i][j].close();

        values.close();
        vectors.close();
    }
}




//    /**
//     * Sets the eiganvalues.
//     *
//     *
//     * @return this
//     */
//    public final Eigen setEigenVals() {
//
//        Kernel.run("eigenValsBatch", handle,
//                size(),
//                mat[0][0], P.to(mat[0][0].ld()),
//                P.to(mat[0][1]), P.to(mat[0][1].ld()),
//                P.to(mat[0][2]), P.to(mat[0][2].ld()),
//                P.to(mat[1][1]), P.to(mat[1][1].ld()),
//                P.to(mat[1][2]), P.to(mat[1][2].ld()),
//                P.to(mat[2][2]), P.to(mat[2][2].ld()),
//                P.to(mat[0][0].entriesPerLine()),
//                P.to(values),
//                P.to(values.ld()),
//                P.to(values.entriesPerLine()),
//                P.to(downsampleFactorXY)
//        );
//
//        return this;
//    }
//
///**
//     * Sets the eiganvectors.
//     *
//     * @param vecsIndex, 0 for the first eigen vector of each matrix, 1 for the
//     * second, or 2 for the third.
//     * @return this
//     */
//    public final Eigen setEiganVectors(int vecsIndex) {
//
//        Kernel.run("eigenVecBatch3x3", handle,
//                size(),
//                mat[0][0], P.to(mat[0][0].ld()),
//                P.to(mat[0][1]), P.to(mat[0][1].ld()),
//                P.to(mat[0][2]), P.to(mat[0][2].ld()),
//                P.to(mat[1][1]), P.to(mat[1][1].ld()),
//                P.to(mat[1][2]), P.to(mat[1][2].ld()),
//                P.to(mat[2][2]), P.to(mat[2][2].ld()),
//                P.to(height),
//                P.to(vectors),
//                P.to(vectors.ld()),
//                P.to(vectors.entriesPerLine()),
//                Pointer.to(values.pointer(vecsIndex)),
//                P.to(values.ld()),
//                P.to(values.entriesPerLine()),
//                P.to(downsampleFactorXY),
//                P.to(vecsIndex),
//                P.to(tolerance)
//        );
//
////        System.out.println("fijiPlugin.Eigan.setEiganVectors() \n" + Arrays.toString(vectors));
//        return this;
//    }
