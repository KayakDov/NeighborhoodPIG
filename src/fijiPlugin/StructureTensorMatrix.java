package fijiPlugin;

import JCudaWrapper.algebra.ColumnMajor;
import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
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
    private final TensorOrd3Stride orientation, coherence;
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
        
        orientation = grad.x().emptyCopyDimensions();
        coherence = grad.x().emptyCopyDimensions();
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
    public MatricesStride setVecs0ToPi() {
        MatricesStride eVecs = eigen.vectors;
        KernelManager.get("vecToNematic").mapToSelf(handle,
                eVecs.dArray(), eVecs.colDist,
                eVecs.getBatchSize() * eVecs.width
        );
        return eVecs;
    }

    /**
     * Sets the orientations from the eigenvectors.
     *
     * @return The orientation matrix.
     */
    public TensorOrd3Stride setOrientations() {
        KernelManager.get("atan2").map(handle, orientation.dArray().length,
                eigen.vectors.dArray(), eigen.vectors.getStrideSize(),
                orientation.dArray(), 1);
        return orientation;
    }

    /**
     * Sets and returns the coherence matrix.
     *
     * @param workSpace Should be the size of the image.
     * @return The coherence matrix.
     */
    public TensorOrd3Stride setCoherence(DArray workSpace) {
        Vector[] l = eigen.values.vecPartition();
        Vector denom = new Vector(handle, workSpace, 1)
                .setSum(1, l[0], 1, l[1]);
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
    public TensorOrd3Stride getOrientations() {
        return orientation;
    }


//        (R, G, B) = (256*cos(x), 256*cos(x + 120), 256*cos(x - 120))  <- this is for 360.  For 180 maybe:
//    (R, G, B) = (256*cos(x), 256*cos(x + 60), 256*cos(x + 120))
    

    /**
     * An array of integers where each value represents a color for a
     * colum-major corresponding orientation.
     *
     * @param single true to map colors to single values, and false to map them to color triplets.
     * @return a column major array of colors. The first 3 elements are the RGB
     * values for the first color, etc...
     */
    public IArray getRGBs(boolean single) {

        setVecs0ToPi();
        setCoherence(orientation.dArray());
        setOrientations().dArray().multiply(handle, 2, 1);
        IArray colors = IArray.empty(orientation.size()*(single?1:3));

        KernelManager.get(single?"colorSingle":"colorTriplet").map(
                handle, 
                orientation.size(),
                orientation.dArray(), 
                IArray.cpuPoint(1),
                colors.pToP(),
                coherence.dArray().pToP(),
                IArray.cpuPoint(1)
        );

        orientation.dArray().multiply(handle, 0.5, 1);
        return colors;

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        strctTensors.close();
        eigen.close();
        orientation.close();
        coherence.close();
    }

    @Override
    public int getColDist() {
        return orientation.colDist;
    }

    /**
     * The eigenvalues and vectors of the structure tensors.
     *
     * @return The eigenvalues and vectors of the structure tensors.
     */
    public Eigen getEigen() {
        return eigen;
    }

}
