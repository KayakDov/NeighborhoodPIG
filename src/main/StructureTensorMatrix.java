package main;

import JCudaWrapper.algebra.ColumnMajor;
import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
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
    private final Matrix orientation, coherence;
    private Handle handle;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param dX The pixel intensity gradient of the image in the x direction.
     * @param dY The pixel intensity gradient of the image in the y direction.
     * @param neighborhoodRad A square window considered a neighborhood around a
     * point. This is the distance from the center of the square to the nearest
     * point on the edge.
     * @param tolerance How close a number be to 0 to be considered 0.
     */
    public StructureTensorMatrix(Matrix dX, Matrix dY, int neighborhoodRad, double tolerance) {

        handle = dX.getHandle();

        int height = dX.getHeight(), width = dX.getWidth();

        strctTensors = new MatricesStride(handle, 2, dX.size());//reset to 3x3 for dZ.

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(dX.getHandle(), neighborhoodRad, height, width)) {
            nps.set(dX, dX, strctTensors.matIndices(0, 0));
            nps.set(dX, dY, strctTensors.matIndices(0, 1));
            nps.set(dY, dY, strctTensors.matIndices(1, 1));

//            nps.set(dX, dZ, strctTensors.get(0, 2));
//            nps.set(dY, dZ, strctTensors.get(1, 2));
//            nps.set(dZ, dZ, strctTensors.get(2, 2));//Add these when working with dZ.
        }

        strctTensors.matIndices(1, 0).set(strctTensors.matIndices(0, 1));
//        strctTensors.get(2, 0).set(strctTensors.get(0, 2)); //engage for 3x3.
//        strctTensors.get(2, 1).set(strctTensors.get(1, 2));

        eigen = new Eigen(strctTensors, tolerance);

        orientation = new Matrix(handle, height, width);
        coherence = new Matrix(handle, height, width);
    }

    /**
     * The tensors are stored in one long row of 2x2 tensors in column major
     * order.
     *
     * @param picRow The row of the pixel in the picture for which the tensor's
     * index is desired.
     * @param picCol The row of the column in the picture for which the tensor's
     * image is desired.
     * @return The index of the beginning of the tensor matrix for the requested
     * pixel.
     */
    private int tensorFirstColIndex(int picRow, int picCol) {

        int tensorSize = strctTensors.height * strctTensors.height;

        return (picCol * orientation.getHeight() + picRow) * tensorSize;
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
        Kernel.run("vecToNematic", handle,
                eVecs.getBatchSize() * eVecs.width,
                eVecs.array(), 
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
    public Matrix setOrientations() {
        Kernel.run("atan2", handle, 
                orientation.size(),
                eigen.vectors.array(), 
                P.to(eigen.vectors.getStrideSize()),
                P.to(orientation), 
                P.to(1)
        );
        return orientation;
    }

    /**
     * Sets and returns the coherence matrix.
     *
     * @param workSpace Should be the size of the image.
     * @return The coherence matrix.
     */
    public Matrix setCoherence(DArray workSpace) {
        Vector[] l = eigen.values.vecPartition();
        Vector denom = new Vector(handle, workSpace, 1)
                .setSum(1, l[0], 1, l[1]);
        Vector coherenceVec = new Vector(handle, coherence.array(), 1)
                .setSum(1, l[0], -1, l[1])
                .ebeDivide(denom);
        coherenceVec.ebeSetProduct(coherenceVec, coherenceVec);
        return coherence;
    }

    /**
     * The coherence matrix.
     * @return The coherence matrix.
     */
    public Matrix getCoherence() {
        return coherence;
    }

    
    
    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public Matrix getOrientations() {
        return orientation;
    }

    /**
     * Takes in a vector, in column major order for a matrix with orientation
     * dimensions, and returns a double[][] representing the vector in
     * [row][column] format.
     *
     * @param columnMajor A vector that is column major order of a matrix with
     * height orientation.height.
     * @param workSpace An auxillery workspace. It should be height in length.
     * @return a cpu matrix.
     */
    private double[][] getRows(Vector columnMajor) {
        return columnMajor.subVectors(1, orientation.getWidth(), orientation.colDist, orientation.getHeight())
                .copyToCPURows();
    }

//        (R, G, B) = (256*cos(x), 256*cos(x + 120), 256*cos(x - 120))  <- this is for 360.  For 180 maybe:
//    (R, G, B) = (256*cos(x), 256*cos(x + 60), 256*cos(x + 120))
    /**
     * Three matrices for red green blue color values.
     *
     * @return a column major array of colors. The first 3 elements are the RGB
     * values for the first color, etc...
     */
    public IArray getRGB() {

        setVecs0ToPi();
        setCoherence(orientation.array());
        setOrientations().multiply(2);

        IArray colors = new IArray(orientation.size() * 3);

        Kernel.run("colors", handle, 
                orientation.size(), 
                orientation.array(), 
                P.to(1), 
                P.to(colors), 
                P.to(3),
                P.to(coherence),
                P.to(1)
        );

        orientation.multiply(0.5);
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
        return orientation.getHeight();
    }

    /**
     * The eigenvalues and vectors of the structure tensors.
     * @return The eigenvalues and vectors of the structure tensors.
     */
    public Eigen getEigen() {
        return eigen;
    }

    /**
     * The data for the structure tensors.
     * @return 
     */
    @Override
    public DArray array() {
        return strctTensors.array();
    }
    
    

}
