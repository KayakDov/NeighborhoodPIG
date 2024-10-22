package main;

import algebra.Matrix;
import algebra.Vector;
import resourceManagement.Handle;

/**
 * This class implements element-by-element multiplication (EBEM) for
 * neighborhood-based matrix operations. It computes the sum of products from
 * neighborhoods in two input matrices, storing the results in the specified
 * vector.
 *
 * The input matrices are expected to have equal dimensions and column distances
 * (colDist).
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodProductSums implements AutoCloseable {

    private final Vector halfNOnes;
    private final Matrix ebeStorage, inRowSum, result;
    private final int nRad, height, width, resultInc;

    /**
     * Constructs a {@code NeighborhoodProductSums} instance to compute the sum
     * of element-by-element products for neighborhoods within two matrices.
     *
     * @param handle A resource handle for creating internal matrices.
     * @param nRad Neighborhood radius; the distance from the center of a
     * neighborhood to its edge.
     * @param height The height of expected matrices.  That is, matrices that will be passed to the set method.
     * @param width The width of expected matrices.
     * @param result A matrix that is one long row of n x n tensors. The height
     * of the matrix is the height of one tensor and the length of the matrix is
     * the size*tensor.length.
     */
    public NeighborhoodProductSums(Handle handle, int nRad, int height, int width, Matrix result) {
        this.nRad = nRad;
        this.height = height;
        this.width = width;
        inRowSum = new Matrix(handle, height, width);
        ebeStorage = new Matrix(handle, height, width);
        halfNOnes = new Vector(handle, nRad + 1).fill(1);
        this.result = result;
        this.resultInc = result.getHeight() * result.getHeight();
    }

    /**
     * Computes neighborhood element-wise multiplication of matrices a and b.
     * Divided into row and column stages for better performance. Then places in
     * result the summation of all the ebe products in the neighborhood of an
     * index pair in that index pair (column major order).
     *
     * @param a The first matrix.
     * @param b The second matrix.
     * @param firstResult The index in the result vector to store the first
     * result.
     */
    public void set(Matrix a, Matrix b, int firstResult) {

        ebeStorage.asVector().mapEbeMultiplyToSelf(a.asVector(), b.asVector());

        inRowSumsEdge();
        inRowSumNearEdge();
        inRowSumCenter();

        MatrixOnVector nSums = new MatrixOnVector(
                result.newDimensions(result.getHeight()*result.getHeight())
                        .getRowVector(firstResult), 
                height, 
                width
        );

        nSumEdge(nSums);
        nSumNearEdge(nSums);
        nSumCenter(nSums);
    }

    /**
     * Handles column summation for the first and last columns (Stage I).
     *
     * @param inRowSum Matrix to store intermediate row sums.
     * @param ebeStorage Element-wise multiplied matrix.
     * @param nRad Neighborhood radius.
     * @param height Height of the matrix.
     * @param width Width of the matrix.
     * @param halfNOnes Vector of ones used for summing the first and last
     * columns.
     */
    private void inRowSumsEdge() {
        inRowSum.getColumnMatrix(0).multiplyAndSet(
                ebeStorage.getSubMatrix(0, height, 0, nRad + 1),
                halfNOnes.vertical()
        );
        inRowSum.getColumnMatrix(width - 1).multiplyAndSet(
                ebeStorage.getSubMatrix(0, height, width - nRad - 1, width),
                halfNOnes.vertical()
        );
    }

    /**
     * Handles column summation for columns near the first and last (Stage II).
     *
     * @param inRowSum Matrix to store intermediate row sums.
     * @param ebeStorage Element-wise multiplied matrix.
     * @param nRad Neighborhood radius.
     * @param width Width of the matrix.
     */
    private void inRowSumNearEdge() {
        for (int i = 1; i < nRad + 1; i++) {
            inRowSum.getColumnMatrix(i).addAndSet(
                    1, ebeStorage.getColumnMatrix(i + nRad),
                    1, inRowSum.getColumnMatrix(i - 1)
            );
            int colInd = width - 1 - i;
            inRowSum.getColumnMatrix(colInd).addAndSet(
                    1, ebeStorage.getColumnMatrix(colInd - nRad),
                    1, inRowSum.getColumnMatrix(colInd + 1)
            );
        }
    }

    /**
     * Handles column summation for the central columns (Stage III).
     *
     * @param inRowSum Matrix to store intermediate row sums.
     * @param ebeStorage Element-wise multiplied matrix.
     * @param nRad Neighborhood radius.
     * @param width Width of the matrix.
     */
    private void inRowSumCenter() {
        for (int colIndex = nRad + 1; colIndex < width - nRad; colIndex++) {
            Matrix inRowNSumsCol = inRowSum.getColumnMatrix(colIndex);
            inRowNSumsCol.addAndSet(
                    -1, ebeStorage.getColumnMatrix(colIndex - nRad - 1),
                    1, ebeStorage.getColumnMatrix(colIndex + nRad)
            );
            inRowNSumsCol.addAndSet(1, inRowNSumsCol, 1, inRowSum.getColumnMatrix(colIndex - 1));
        }
    }

    /**
     * Handles row summation for the first and last rows (Stage I).
     *
     * @param nSums Matrix on the result vector to store the results.
     * @param inRowSum Matrix containing intermediate row sums.
     * @param nRad Neighborhood radius.
     * @param width Width of the matrix.
     * @param halfNOnes Vector of ones used for summing the first and last rows.
     */
    private void nSumEdge(MatrixOnVector nSums) {
        nSums.getRow(0).multiplyAndSet(
                halfNOnes.horizontal(),
                inRowSum.getSubMatrix(0, nRad + 1, 0, width)
        );
        nSums.getRow(nSums.height - 1).multiplyAndSet(
                halfNOnes.horizontal(),
                inRowSum.getSubMatrix(nSums.height - nRad - 1, nSums.height, 0, width)
        );
    }

    /**
     * Handles row summation for rows near the first and last (Stage II).
     *
     * @param nSums Matrix on the result vector to store the results.
     * @param inRowSum Matrix containing intermediate row sums.
     * @param nRad Neighborhood radius.
     * @param height Height of the matrix.
     */
    private void nSumNearEdge(MatrixOnVector nSums) {
        for (int i = 1; i < nRad + 1; i++) {
            int rowInd = i;
            nSums.getRow(rowInd).addAndSet(
                    1, inRowSum.getRowMatrix(rowInd + nRad),
                    1, nSums.getRow(rowInd - 1)
            );
            rowInd = height - 1 - i;
            nSums.getRow(rowInd).addAndSet(
                    1, inRowSum.getRowMatrix(rowInd - nRad),
                    1, nSums.getRow(rowInd + 1)
            );
        }
    }

    /**
     * Handles row summation for the central rows (Stage III).
     *
     * @param nSums Matrix on the result vector to store the results.
     * @param inRowSum Matrix containing intermediate row sums.
     * @param nRad Neighborhood radius.
     * @param height Height of the matrix.
     */
    private void nSumCenter(MatrixOnVector nSums) {
        for (int rowIndex = nRad + 1; rowIndex < height - nRad; rowIndex++) {
            Matrix nSumsRow = nSums.getRow(rowIndex);
            nSumsRow.addAndSet(
                    -1, inRowSum.getRowMatrix(rowIndex - nRad - 1),
                    1, inRowSum.getRowMatrix(rowIndex + nRad)
            );
            nSumsRow.addAndSet(1, nSumsRow, 1, nSums.getRow(rowIndex - 1));
        }
    }

    /**
     * This class provides a wrapper around a {@link Vector} to access its data
     * as a matrix-like structure. It allows retrieving rows of the vector as
     * submatrices, facilitating matrix-like operations on a linear structure.
     * The vector is interpreted as a 2D matrix with specified dimensions.
     *
     * This is useful when working with element-wise matrix operations and the
     * results of those operations are stored in a {@link Vector} but need to be
     * accessed as a matrix.
     *
     * @author E. Dov Neimand
     */
    private static class MatrixOnVector {

        /**
         * The underlying vector storing the matrix data.
         */
        private final Vector vec;

        /**
         * The height (number of rows) of the matrix.
         */
        private final int height;

        /**
         * The width (number of columns) of the matrix.
         */
        private final int width;

        /**
         * Constructs a {@code MatrixOnVector} instance to interpret a vector as
         * a matrix with the given dimensions.
         *
         * @param vec The vector storing the matrix elements.
         * @param height The number of rows in the matrix.
         * @param width The number of columns in the matrix.
         */
        public MatrixOnVector(Vector vec, int height, int width) {
            this.vec = vec;
            this.height = height;
            this.width = width;
        }

        /**
         * Retrieves the specified row from the vector, interpreting the data as
         * a matrix. The row is returned as a submatrix (horizontal).
         *
         * @param rowIndex The index of the row to retrieve.
         * @return A {@link Matrix} representing the row at the specified index.
         */
        public Matrix getRow(int rowIndex) {
            return vec.getSubVector(rowIndex, width, height).horizontal();
        }
    }

    /**
     * Cleans up allocated memory on the gpu.
     */
    @Override
    public void close() {
        halfNOnes.close();
        ebeStorage.close();
        inRowSum.close();
    }
}
