//package main;
//
//
//
//import algebra.Matrix;
//import algebra.MatrixBatch;
//import algebra.Vector;
//import resourceManagement.Handle;
//import array.DArray;
//
///**
// * The {@code GradientRows} class stores gradient information for neighborhoods
// * of pixels from an image. Each row in the {@code gradientRows} matrix contains
// * the gradient values for a all the pixel neighborhood of a row in the original
// * image, where the neighborhood size is determined by the
// *
// * <p>
// * This class implements {@link AutoCloseable} to ensure proper resource
// * management.</p>
// *
// * @author E. Dov Neimand
// */
//public class GradientNeighborRows implements AutoCloseable {
//
//    /**
//     * A column of gradients. Each even column c is all the x gradients near row
//     * c, neighborhood column stakced on neighborhood column, and similarly for
//     * odd columns and y rows.
//     */
//    private final Matrix gradientRowsAsColumns;
//    private final int neighborhoodWidth;
//    private final int height, width;
//
//
//    /**
//     * The constructor.
//     *
//     * @param dX the matrix containing the gradients in the x direction
//     * @param dY the matrix containing the gradients in the y direction
//     * @param n the height and width of the neighborhood around each pixel to
//     * consider
//     */
//    public GradientNeighborRows(Matrix dX, Matrix dY, int n) {
//        height = dX.getHeight();
//        width = dX.getWidth();
//    }
//    
//    
//    
//    
//    public GradientNeighborRows(Matrix dX, Matrix dY, int n) {
//        width = dX.getWidth();
//        height = dX.getHeight();
//        nColHeight = (2 * (n / 2) + 1);
//
//        this.neighborhoodWidth = n;
//
//        gradientRowsAsColumns = new Matrix(dX.getHandle(),
//                width * nColHeight + (n / 2) * 2,
//                height * 2).fill(0);
//
//        for (int picRow = 0; picRow < dX.getHeight(); picRow++) {
//            int startRow = startRow(picRow), endRow = endRow(picRow);
//
//            Matrix neighborhoodX = dX.getSubMatrix(startRow, endRow, 0, width);
//            Matrix neighborhoodY = dY.getSubMatrix(startRow, endRow, 0, width);
//
//            for (int picCol = 0; picCol < width; picCol++) {
//                getXCol(picRow).setSubVector(nColHeight * picCol + n / 2 * nColHeight, neighborhoodX.getColumnVector(picCol));
//                getYCol(picRow).setSubVector(nColHeight * picCol + n / 2 * nColHeight, neighborhoodY.getColumnVector(picCol));
//            }
//
//        }
//    }
//    private final int nColHeight,
//    /**
//     * Determines the starting row index for a given pixel row, ensuring that
//     * the row does not exceed the image boundaries.
//     *
//     * @param picRow the pixel row in the original image
//     * @return the starting row index for the neighborhood
//     */
//    private int startRow(int picRow) {
//        return Math.max(picRow - neighborhoodWidth / 2, 0);
//    }
//
//    /**
//     * Determines the ending row index for a given pixel row, ensuring that the
//     * row does not exceed the image boundaries.
//     *
//     * @param picRow the pixel row in the original image
//     * @return the ending row index for the neighborhood
//     */
//    private int endRow(int picRow) {
//        return Math.min(picRow + neighborhoodWidth / 2 + 1, gradientRowsAsColumns.getHeight());
//    }
//
//    /**
//     * Retrieves the x-direction gradient row for a specified pixel row.
//     *
//     * @param picRow the pixel row in the original image
//     * @return the vector representing the x-direction gradients for the row
//     */
//    private Vector getXRow(int picRow) {
//        return gradientRowsAsColumns.getRowVector(picRow * 2);
//    }
//
//    /**
//     * Retrieves the y-direction gradient row for a specified pixel row.
//     *
//     * @param picRow the pixel row in the original image
//     * @return the vector representing the y-direction gradients for the row
//     */
//    private Vector getYRow(int picRow) {
//        return gradientRowsAsColumns.getRowVector(picRow * 2 + 1);
//    }
//
//    /**
//     * Retrieves the x-direction gradient col for a specified pixel row.
//     *
//     * @param picCol the pixel row in the original image
//     * @return the vector representing the x-direction gradients for the row
//     */
//    private Vector getXCol(int picCol) {
//        return gradientRowsAsColumns.getColumnVector(picCol * 2);
//    }
//
//    /**
//     * Retrieves the y-direction gradient column for a specified pixel row.
//     *
//     * @param picRow the pixel row in the original image
//     * @return the vector representing the y-direction gradients for the row
//     */
//    private Vector getYCol(int picCol) {
//        return gradientRowsAsColumns.getColumnVector(2 * picCol + 1);
//    }
//
//    /**
//     * The matrix of structure tensors.
//     *
//     * This is a 2x2*pic.size() matrix. Each 2x2 submatrix is in column major
//     * order corresponding to the column major order of the underlying picture.
//     *
//     *
//     * @param handle The handle.
//     * @return The matrix of structure tensors.
//     */
//    public Matrix structureTensorMatrix(Handle handle) {
//        Matrix structureTensors = new Matrix(handle, 2, height * width * 2);  //A row of tensor matrices because that's a format for computing eigan values.
//
//        MatrixBatch result = new MatrixBatch(structureTensors, 2, 2, 2, 2);
//        MatrixBatch neighborGradsRow = new MatrixBatch(gradientRowsAsColumns, nColHeight, 2, nColHeight * nColHeight, 2);
//        MatrixBatch neighborGradsTranspose = neighborGradsRow.shallowCopy();
//        neighborGradsTranspose.transpose();
//
//        result.addToMeMatMatMult(handle, 1, neighborGradsRow, neighborGradsTranspose, 0);
//
//        return structureTensors;
//    }
//
//    /**
//     * Closes the underlying gradient matrix resources. This method is called to
//     * release the memory allocated for the gradient matrix when it is no longer
//     * needed.
//     *
//     */
//    @Override
//    public void close() {
//        gradientRowsAsColumns.close();
//    }
//}
