package main;

import MathSupport.Rotation;
import algebra.Matrix;
import algebra.Vector;
import array.DArray2d;
import static java.lang.Math.*;
import javax.print.attribute.standard.OrientationRequested;
import resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrix implements AutoCloseable {

    /**
     * This matrix is a row of tensors. The height of this matrix is the height
     * of one tensor and the length of this matrix is the number of pixels times
     * the length of a tensor. The order of the tensors is column major, that
     * is, the first tensor corresponds to the pixel at column 0 row 0, the
     * second tensor corresponds to the pixel at column 0 row 1, etc...
     */
    private Matrix strctTensors;

    private Matrix strTenEVecs;
    private Matrix strTenEVals;
    private Matrix orientation;

    public StructureTensorMatrix(Matrix dX, Matrix dY, int neighborhoodRad) {

        Handle handle = dX.getHandle();
        int height = dX.getHeight(), width = dX.getWidth();

        strctTensors = new Matrix(handle, 2, 2 * dX.size());//reset to 3 for dZ.

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(dX.getHandle(), neighborhoodRad, height, width, strctTensors)) {
            nps.set(dX, dX, 0);
            nps.set(dX, dY, 1);
            nps.set(dY, dY, 3);//Reset these when working with dZ.
        }

        int numElementsPerTensor = strctTensors.getHeight() * strctTensors.getHeight();

        Vector copyTo = strctTensors.asVector().getSubVector(2, dX.size(), numElementsPerTensor);
        Vector copyFrom = strctTensors.asVector().getSubVector(1, dX.size(), numElementsPerTensor);

        copyTo.set(copyFrom);

        orientation = new Matrix(dX.getHandle(), dX.getHeight(), dX.getWidth());
        setEigans();

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

        int tensorSize = strctTensors.getHeight() * strctTensors.getHeight();

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

        int height = strctTensors.getHeight(), size = height * height;

        int startCol = col * orientation.getHeight() * strctTensors.getHeight();

        return strctTensors.getSubMatrix(0, height, startCol, startCol + height);
    }

    /**
     * Retrieves an eigenvector.
     *
     * @param row The row of the pixel for which the eigenvector is desired.
     * @param col The column of the pixel for which the eigenvector is desired.
     * @param eiganIndex If there is more than one eigenvector, the index of the
     * one you want.
     * @return The requested eigenvector.
     */
    public Vector getEiganVec(int row, int col, int eiganIndex) {
        return strTenEVecs.getColumnVector(tensorFirstColIndex(row, col) + eiganIndex);
    }

    /**
     * Retrieves an eigenvalue.
     *
     *
     * @param row The row of the pixel for which the eigenvalue is desired.
     * @param col The column of the pixel for which the eigenvalue is desired.
     * @param eiganIndex If there is more than one eigenvalue, the index of the
     * one you want.
     * @return The requested eigenvalue.
     */
    public Vector getEiganVal(int row, int col, int eiganIndex, int tensorSize) {
        return strTenEVals.getColumnVector(tensorFirstColIndex(row, col) + eiganIndex);
    }

    /**
     * sets the orientations.
     */
    private void setEigans() {
        Handle handle = strctTensors.getHandle();

        int numTensors = strctTensors.getWidth() / strctTensors.getHeight();

        strTenEVecs = new Matrix(handle, strctTensors.getHeight(), strctTensors.getWidth());
        strTenEVals = new Matrix(handle, 1, strctTensors.getWidth());

        DArray2d.computeEigen(
                strctTensors.getHeight(),
                strctTensors.dArray(), strctTensors.getHeight(),
                strTenEVals.dArray(),
                strTenEVecs.dArray(), strctTensors.getHeight(),
                numTensors,
                DArray2d.Fill.FULL
        );

        strctTensors.getHandle().synch();
    }

    /**
     * Sets the orientations from the eigenvectors.
     */
    public void setOrientations() {
        orientation.dArray().atan2(strTenEVecs.dArray());
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public Matrix getOrientations() {
        return orientation.unmodifable();
    }

    /**
     * Gives the cos of the angle each 2d column vector in the matrix forms with
     * the x axis.
     *
     * @param eachColIsVec2d A matrix where each column is a 2d vector.
     * @return A vector where each index the cos of each column in m.
     */
    private Vector cosOf(double alpha, Matrix eachColIsVec2d, Matrix rotate) {
        Matrix rotated = rotate.multiply(eachColIsVec2d);
        Vector cosDenominator = new Vector(eachColIsVec2d.getHandle(), eachColIsVec2d.getWidth());
        DArray2d.multMatMatStridedBatched(eachColIsVec2d.getHandle(),
                true, false,
                2, 1, 1,
                alpha,
                eachColIsVec2d.dArray(), 2, 2,
                eachColIsVec2d.dArray(), 2, 2,
                0, cosDenominator.dArray(), 1, 1,
                cosDenominator.getDimension());

        Vector xVals = new Vector(eachColIsVec2d.getHandle(), eachColIsVec2d.dArray(), eachColIsVec2d.getColDist());

        return xVals.ebeDivide(cosDenominator);
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

        try (Matrix red = cosOf(256, strTenEVals, Rotation.id).asMatrix(orientation.getHeight())) {
            RGB[0] = red.getData();
        }
        try (Matrix green = cosOf(256, strTenEVals, Rotation.r60).asMatrix(orientation.getHeight())) {
            RGB[1] = green.getData();
        }
        try (Matrix blue = cosOf(256, strTenEVals, Rotation.r60).asMatrix(orientation.getHeight())) {
            RGB[2] = blue.getData();
        }

        return RGB;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        strctTensors.close();
        strTenEVecs.close();
        strTenEVals.close();
        orientation.close();
    }

}
