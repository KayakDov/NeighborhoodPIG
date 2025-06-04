package fijiPlugin;

import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.runtime.JCuda;
import main.Test;

/**
 * A set of 3x3 matrices, their eigenvectors and values.
 *
 * @author E. Dov Neimand
 */
public class Eigen implements AutoCloseable {//TODO: maybe incorporate this up into StructureTensorMatrices.

    private final double tolerance;
    private final int downsampleFactorXY;
    private final Dimensions dim;
    private final Handle handle;

    /**
     * Each layer in this matrix is for a different pixel, in column major
     * order.
     */
    public PArray2dToD2d[][] mat;

//    /**
//     * These are organized in the same columns, layers, and grids as the initial
//     * picture. The rows are changed so that each set of 3 eigenvectors are
//     * consecutive in each column.
//     */
//    public final PArray2dToD2d values;//TODO: there may not be any need to store eigenvalues.
//
//    /**
//     * The first eigevector of each structureTensor with mathcin columns and
//     * layers as the original pixels, and rows * 3.
//     */
//    public final PArray2dToD2d vectors;
    /**
     *
     * @param handle
     * @param dim
     * @param downSampleFactorXY 1 in every how many pixels get evaluated in the
     * x and y dimensions.
     * @param tolerance
     */
    public Eigen(Handle handle, Dimensions dim, int downSampleFactorXY, double tolerance) {
        this.dim = dim;
        this.handle = handle;
        this.downsampleFactorXY = downSampleFactorXY;
        this.tolerance = tolerance;

        int numDim = dim.hasDepth() ? 3 : 2;
        mat = new PArray2dToD2d[numDim][numDim];

        for (int i = 0; i < numDim; i++)
            for (int j = i; j < numDim; j++)
                mat[i][j] = mat[j][i] = dim.emptyP2dToD2d(handle);
    }

    /**
     * All the values of all the structure tensors at the given indices.
     *
     * @param row The row of the desired vector.
     * @param col The column of the desired vector.
     * @return All the values of all the structure tensors at the given indices.
     */
    public PArray2dToD2d getMatValsAt(int row, int col) {
        return mat[row][col];
    }

    /**
     * Sets the eigen values and the eigen vectors at the requested index.
     *
     * @param eigenInd If the index is 0, then the eigen vectors that corespond
     * to the greatest eigenvalue, if the index is one, then the 2nd greatest.
     * @param vectors
     * @param coherence Where to store the coherence data.
     * @param azimuth Location to load the azimuthal angles.
     * @param zenith Location to load the zenith angles.
     * @param downSampledDim
     * @return this.
     */
    public Eigen set(int eigenInd, PArray2dToF2d vectors, PArray2dToF2d coherence, PArray2dToF2d azimuth, PArray2dToF2d zenith, Dimensions downSampledDim) {
        
        if (dim.hasDepth()) Kernel.run("eigenBatch3d", handle,
                    azimuth.deepSize(),
                    new PArray2dTo2d[]{
                        mat[0][0],
                        mat[0][1],
                        mat[0][2],
                        mat[1][1],
                        mat[1][2],
                        mat[2][2],
                        vectors,
                        coherence,
                        azimuth,
                        zenith
                    },
                    downSampledDim, //TODO: fix cu so that it used downSampledDim
                    P.to(eigenInd),
                    P.to(tolerance)
            );
        else Kernel.run("eigenBatch2d", handle,
                    azimuth.deepSize(),
                    new PArray2dTo2d[]{
                        mat[0][0],
                        mat[0][1],
                        mat[1][1],
                        vectors,
                        coherence,
                        azimuth
                    },
                    downSampledDim,
                    P.to(downsampleFactorXY),
                    P.to(eigenInd),
                    P.to(tolerance)
            );
                
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        if (mat != null)
            for (int i = 0; i < mat.length; i++)
                for (int j = i; j < mat[0].length; j++)
                    mat[i][j].close();
        mat = null;
    }
}