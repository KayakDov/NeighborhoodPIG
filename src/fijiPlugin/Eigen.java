package fijiPlugin;

import FijiInput.UsrInput;
import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.resourceManagement.Handle;

/**
 * A set of 3x3 matrices, their eigenvectors and values.
 *
 * @author E. Dov Neimand
 */
public class Eigen implements AutoCloseable {//TODO: maybe incorporate this up into StructureTensorMatrices.

    private final UsrInput ui;
    private final Dimensions dim;
    private final Handle handle;

    /**
     * Each layer in this matrix is for a different pixel, in column major
     * order.
     */
    public PArray2dToD2d[][] mat;

    /**
     *
     * @param handle
     * @param dim
     * @param ui User defined preferences.  Will be used for down sample factors and tolerance.
     */
    public Eigen(Handle handle, Dimensions dim, UsrInput ui) {
        this.dim = dim;
        this.handle = handle;
        this.ui = ui;

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
    public Eigen set(int eigenInd, P2dToF2d vectors, P2dToF2d coherence, P2dToF2d azimuth, P2dToF2d zenith, Dimensions downSampledDim){
        
//        System.out.println("fijiPlugin.Eigen.set() dim = " + downSampledDim);
        
        if (dim.hasDepth()) handle.runKernel("eigenBatch3d", 
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
                    downSampledDim,
                    P.to(ui.downSampleFactorXY),
                    P.to(ui.downSampleFactorZ.get()),
                    P.to(eigenInd),
                    P.to(ui.tolerance)
            );
        else handle.runKernel("eigenBatch2d", 
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
                    P.to(ui.downSampleFactorXY),
                    P.to(eigenInd),
                    P.to(ui.tolerance)
            );
        
//        System.out.println("fijiPlugin.Eigen.set() Vectors\n" + vectors);
//        System.out.println("fijiPlugin.Eigen.set() azimuth\n" + azimuth);
//        System.out.println("fijiPlugin.Eigen.set() zenith\n" + zenith);
//        System.out.println("fijiPlugin.Eigen.set() coherence\n" + Test.format(coherence.toString()));
//
//        if(true) throw new RuntimeException("TODO:delet me");
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