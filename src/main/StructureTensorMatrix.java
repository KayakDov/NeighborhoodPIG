package main;

import JCudaWrapper.algebra.ColumnMajor;
import MathSupport.Rotation;
import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.DArray;
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
    private Handle handle;

    public StructureTensorMatrix(Matrix dX, Matrix dY, int neighborhoodRad) {

        handle = dX.getHandle();
                
        int height = dX.getHeight(), width = dX.getWidth();

        strctTensors = new MatricesStride(handle, 2, dX.size());//reset to 3x3 for dZ.
        
        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(dX.getHandle(), neighborhoodRad, height, width)) {            
            nps.set(dX, dX, strctTensors.get(0, 0));
            nps.set(dX, dY, strctTensors.get(0, 1));//TODO: Check other arrays for using length as the number of times something is done when increment says it should not be!
            nps.set(dY, dY, strctTensors.get(1, 1));
            
//            nps.set(dX, dZ, strctTensors.get(0, 2));
//            nps.set(dY, dZ, strctTensors.get(1, 2));
//            nps.set(dZ, dZ, strctTensors.get(2, 2));//Add these when working with dZ.
        }
        
        strctTensors.get(1, 0).set(strctTensors.get(0, 1));
//        strctTensors.get(2, 0).set(strctTensors.get(0, 2)); //engage for 3x3.
//        strctTensors.get(2, 1).set(strctTensors.get(1, 2));

        

        eigen = new Eigen(strctTensors);
        
        orientation = new Matrix(handle, height, width);
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
     * Sets the orientations from the eigenvectors.
     * @return The orientation matrix.
     */
    public Matrix setOrientations() {

        KernelManager.get("atan2").map(handle, eigen.vectors.dArray(), 4, orientation.dArray(), 1, orientation.size());
        return orientation;
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
     * @param columnMajor A vector that is column major order of a matrix with height orientation.height.
     * @param workSpace An auxillery workspace.  It should be height in length.
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
     * @return
     */
    public double[][][] getRGB() {

        double RGB[][][] = new double[3][][];

        try (DArray workSpace = DArray.empty(2 * orientation.size())) {
            
            VectorsStride orientationVecs255 = eigen.vectors.column(0).setVectorMagnitudes(255, workSpace);
            
            MatricesStride rotate120 = Rotation.r120.repeating(orientationVecs255.getBatchSize());

            Vector cos = orientationVecs255.get(0);
            RGB[0] = getRows(cos);
            
            orientationVecs255.setProduct(rotate120, orientationVecs255);
            RGB[1] = getRows(cos);
            
            orientationVecs255.setProduct(rotate120, orientationVecs255);
            RGB[2] = getRows(cos);
            
            orientationVecs255.setProduct(rotate120, orientationVecs255);
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
