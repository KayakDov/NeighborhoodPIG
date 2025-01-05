package fijiPlugin;

import JCudaWrapper.algebra.ColumnMajor;
import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrix implements AutoCloseable, ColumnMajor {

    /**
     * This matrix is a row of tensors. The height of this matrix is the height
     * of one tensor and the length of this matrix is the number of pixels times
     * the length of a tensor. The order of the tensors is column major, that
     * is, the first tensor corresponds to the pixel at column 0 row 0, the
     * second tensor corresponds to the pixel at column 0 row 1, etc...
     */
    private final MatricesStride strctTensors;

    private final Eigen eigen;
    private final TensorOrd3Stride orientationXY, orientationYZ, coherence;
    private Handle handle;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param grad The pixel intensity gradient of the image.
     * @param neighborhoodRad A square window considered a neighborhood around a
     * point. This is the distance from the center of the square to the nearest
     * point on the edge.
     * @param tolerance How close a number be to 0 to be considered 0.
     */
    public StructureTensorMatrix(Gradient grad, int neighborhoodRad, double tolerance) {

        handle = grad.x().handle;

        strctTensors = new MatricesStride(handle, 3, grad.size());

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, neighborhoodRad, grad.height, grad.width, grad.depth, grad.batchSize)) {
            nps.set(grad.x(), grad.x(), strctTensors.matIndices(0, 0));
            nps.set(grad.x(), grad.y(), strctTensors.matIndices(0, 1));
            nps.set(grad.y(), grad.y(), strctTensors.matIndices(1, 1));
            nps.set(grad.x(), grad.z(), strctTensors.matIndices(0, 2));
            nps.set(grad.y(), grad.z(), strctTensors.matIndices(1, 2));
            nps.set(grad.z(), grad.z(), strctTensors.matIndices(2, 2));
        }

        strctTensors.matIndices(1, 0).set(strctTensors.matIndices(0, 1));
        strctTensors.matIndices(2, 0).set(strctTensors.matIndices(0, 2));
        strctTensors.matIndices(2, 1).set(strctTensors.matIndices(1, 2));

        eigen = new Eigen(strctTensors, tolerance);

        orientationXY = grad.x().emptyCopyDimensions();
        orientationYZ = grad.x().emptyCopyDimensions();
        coherence = grad.x().emptyCopyDimensions();
        
        setVecs0ToPi();
        setCoherence(orientationXY.dArray());
        setOrientations();
    }

    /**
     * Gets the structure tensor from pixel at the given row and column of the
     * picture.
     *
     * @param row The row of the desired pixel.
     * @param col The column of the desired pixel.
     * @return The structure tensor for the given row and column.
     */
    public Matrix getTensor(int row, int col) {

        return strctTensors.getMatrix(index(row, col));
    }

    /**
     * All the eigen vectors with y less than 0 are mulitplied by -1.
     *
     * @return The eigenvectors.
     */
    public final MatricesStride setVecs0ToPi() {
        MatricesStride eVecs = eigen.vectors;
        Kernel.run("vecToNematic", handle,
                eVecs.getBatchSize() * eVecs.width,
                eVecs.dArray(),
                P.to(eVecs.colDist),
                P.to(eVecs),
                P.to(eVecs.colDist)
        );
        return eVecs;
    }

    /**
     * Sets the orientations from the eigenvectors.
     *
     * @return The orientation matrix.
     */
    public final void setOrientations() {
        Kernel.run("atan2", handle,
                orientationXY.dArray().length,
                eigen.vectors.dArray(),
                P.to(eigen.vectors.getStrideSize()),
                P.to(orientationXY),
                P.to(1)
        );
        Kernel.run("atan2", handle,
                orientationXY.dArray().length,
                eigen.vectors.dArray().subArray(1),
                P.to(eigen.vectors.getStrideSize()),
                P.to(orientationYZ),
                P.to(1)
        );
    }

    /**
     * Sets and returns the coherence matrix.
     *
     * @param workSpace Should be the size of the image.
     * @return The coherence matrix.
     */
    public final TensorOrd3Stride setCoherence(DArray workSpace) {
        Vector[] l = eigen.values.vecPartition();
        Vector denom = new Vector(handle, workSpace, 1)
                .setSum(1, l[0], 1, l[1])
                .add(1, l[2]);
        
        Vector coherenceVec = new Vector(handle, coherence.dArray(), 1)
                .setSum(1, l[0], -1, l[1])
                .ebeDivide(denom);
        coherenceVec.ebeSetProduct(coherenceVec, coherenceVec);
        return coherence;
    }

    /**
     * The coherence matrix.
     *
     * @return The coherence matrix.
     */
    public TensorOrd3Stride getCoherence() {
        return coherence;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public TensorOrd3Stride getOrientationXY() {
        return orientationXY;
    }
    
    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public TensorOrd3Stride getOrientationYZ() {
        return orientationYZ;
    }

    

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        strctTensors.close();
        eigen.close();
        orientationXY.close();
        orientationYZ.close();
        coherence.close();
    }

    @Override
    public int getColDist() {
        return orientationXY.colDist;
    }

    /**
     * The eigenvalues and vectors of the structure tensors.
     *
     * @return The eigenvalues and vectors of the structure tensors.
     */
    public Eigen getEigen() {
        return eigen;
    }

    /**
     * The data for the array of structure tensors.
     * @return 
     */
    @Override
    public DArray dArray() {
        return strctTensors.dArray();
    }

}
