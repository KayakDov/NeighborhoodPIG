package main;

import JCudaWrapper.algebra.ColumnMajor;
import MathSupport.Rotation;
import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.KernelManager;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

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
    private final Matrix orientation;
    private final Handle handle;

    public StructureTensorMatrix(Matrix dX, Matrix dY, int neighborhoodRad) {

        handle = dX.getHandle();
        int height = dX.getHeight(), width = dX.getWidth();

        strctTensors = new MatricesStride(handle, 2, dX.size()).fill(0);//reset to 3x3 for dZ.

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(dX.getHandle(), neighborhoodRad, height, width)) {
            nps.set(dX, dX, strctTensors.get(0, 0));
            nps.set(dX, dY, strctTensors.get(0, 1));
            nps.set(dY, dY, strctTensors.get(1, 1));
//            nps.set(dX, dZ, 3);
//            nps.set(dY, dZ, 3);
//            nps.set(dZ, dZ, 3);//Add these when working with dZ.
        }
                
        
        strctTensors.get(1, 0).set(strctTensors.get(0, 1));
//        strctTensors.get(2, 0).set(strctTensors.get(0, 2)); //engage for 3x3.
//        strctTensors.get(2, 1).set(strctTensors.get(1, 2));

        orientation = new Matrix(handle, height, width);
        eigen = new Eigen(strctTensors, false);//set to true for 3x3.

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

        return strctTensors.getSubMatrix(index(row, col));
    }


    /**
     * Sets the orientations from the eigenvectors.
     */
    public void setOrientations() {
        KernelManager.get("atan2").map(handle, eigen.vectors.dArray(), 4, orientation.dArray(), 1, orientation.size());
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
     * Gives the cos of the angle each 2d column vector in the matrix forms with
     * the x axis.
     *
     * @param vecs2d A matrix where each column is a 2d vector.
     * @return A vector where each index the cos of each column in m.
     */
    private Vector cosOf(double alpha, VectorsStride vecs2d, Matrix rotate) {
        
        
        VectorsStride rotated = new VectorsStride(handle, 2, vecs2d.dArray().batchSize, vecs2d.getSubVecDim(), 1);
        rotated.setMatVecMult(
                rotate.repeating(vecs2d.dArray().batchCount()), 
                vecs2d
        );
        
        Vector xVals = vecs2d.getElement(0);
        
        Vector cosDenominator = new Vector(handle, vecs2d.dArray().batchSize);
        
        cosDenominator.setBatchVecVecMult(vecs2d, vecs2d);
        
        cosDenominator.mapEBEDivide(xVals);

        return xVals;
    }

//        (R, G, B) = (256*cos(x), 256*cos(x + 120), 256*cos(x - 120))  <- this is for 360.  For 180 maybe:
//    (R, G, B) = (256*cos(x), 256*cos(x + 60), 256*cos(x + 120))
    /**
     * Three matrices for red green blue color values.
     *
     * @return
     */
    public double[][][] getRGB() {

        double RGB[][][] = new double[3][][];

        VectorsStride primaryAxis = eigen.vectors.column(0);
        
        try (Matrix red = cosOf(256, primaryAxis, Rotation.id).asMatrix(orientation.getHeight())) {
            RGB[0] = red.get();
        }
        try (Matrix green = cosOf(256, primaryAxis, Rotation.r60).asMatrix(orientation.getHeight())) {
            RGB[1] = green.get();
        }
        try (Matrix blue = cosOf(256, primaryAxis, Rotation.r60).asMatrix(orientation.getHeight())) {
            RGB[2] = blue.get();
        }

        return RGB;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        strctTensors.close();
        eigen.close();
        orientation.close();
    }

    @Override
    public int getColDist() {
        return orientation.getHeight();
    }

}
