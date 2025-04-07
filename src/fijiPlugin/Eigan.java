package fijiPlugin;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Int.IStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;

/**
 * A set of 3x3 matrices, their eigenvectors and values.
 *
 * @author E. Dov Neimand
 */
public class Eigan extends Dimensions implements AutoCloseable {//TODO fix spelling to eigen

    private final float tolerance;
    private final int downsampleFactorXY;

    /**
     * Each layer in this matrix is for a different pixel, in column major
     * order.
     */
    public final FStrideArray3d[][] mat;

    /**
     * These are organized in the same columns, layers, and grids as the initial
     * picture. The rows are changed so that each set of 3 eigenvectors are
     * consecutive in each column.
     */
    public final FStrideArray3d values;

    /**
     * The first eigevector of each structureTensor with mathcin columns and
     * layers as the original pixels, and rows * 3.
     */
    public final FStrideArray3d[] vectors;

    /**
     *
     * @param handle
     * @param dim
     * @param downSampleFactorXY 1 in every how many pixels get evaluated in the
     * x and y dimensions.
     * @param tolerance
     */
    public Eigan(Handle handle, Dimensions dim, int downSampleFactorXY, float tolerance) {
        super(dim);
        this.downsampleFactorXY = downSampleFactorXY;
        this.tolerance = tolerance;
        mat = new FStrideArray3d[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++)
                mat[j][i] = mat[i][j] = dim.empty();

        values = new FStrideArray3d((dim.height / downSampleFactorXY) * 3, dim.width / downSampleFactorXY, dim.depth, dim.batchSize);
        vectors = new FStrideArray3d[]{values.copyDim(), values.copyDim(), values.copyDim()};

    }

    /**
     * All the values of all the structure tensors at the given indices.
     *
     * @param row The row of the desired vector.
     * @param col The column of the desired vector.
     * @return All the values of all the structure tensors at the given indices.
     */
    public FStrideArray3d at(int row, int col) {
        return mat[row][col];
    }

    /**
     * Sets the eiganvalues.
     *
     *
     * @return this
     */
    public final Eigan setEigenVals() {

        Kernel.run("eigenValsBatch", handle,
                size(),
                mat[0][0], P.to(mat[0][0].ld()),
                P.to(mat[0][1]), P.to(mat[0][1].ld()),
                P.to(mat[0][2]), P.to(mat[0][2].ld()),
                P.to(mat[1][1]), P.to(mat[1][1].ld()),
                P.to(mat[1][2]), P.to(mat[1][2].ld()),
                P.to(mat[2][2]), P.to(mat[2][2].ld()),
                P.to(mat[0][0].entriesPerLine()),
                P.to(values),
                P.to(values.ld()),
                P.to(values.entriesPerLine()),
                P.to(tolerance),
                P.to(downsampleFactorXY)
        );

        return this;
    }

    /**
     * Sets the eiganvectors.
     *
     * @return this
     */
    public final Eigan setEiganVectors() {

        for (int i = 0; i < 3; i++)
            Kernel.run("eigenVecBatch3x3", handle,
                    size(),
                    mat[0][0], P.to(mat[0][0].ld()),
                    P.to(mat[0][1]), P.to(mat[0][1].ld()),
                    P.to(mat[0][2]), P.to(mat[0][2].ld()),
                    P.to(mat[1][1]), P.to(mat[1][1].ld()),
                    P.to(mat[1][2]), P.to(mat[1][2].ld()),
                    P.to(mat[2][2]), P.to(mat[2][2].ld()),
                    P.to(height),

                    P.to(vectors[i]),
                    P.to(vectors[i].ld()),
                    P.to(vectors[i].entriesPerLine()),

                    Pointer.to(values.pointer(i)),
                    P.to(values.ld()),
                    P.to(values.entriesPerLine()),

                    P.to(downsampleFactorXY),
                    P.to(i),
                    P.to(tolerance)
            );

//        System.out.println("fijiPlugin.Eigan.setEiganVectors() \n" + Arrays.toString(vectors));
        
        return this;
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
        Arrays.stream(vectors).forEach(a -> a.close());
    }
}
