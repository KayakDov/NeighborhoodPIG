package main;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.resourceManagement.Handle;

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
    private final Matrix ebeStorage, inRowSum;    
    private final int nRad, height, width;

    /**
     * Constructs a {@code NeighborhoodProductSums} instance to compute the sum
     * of element-by-element products for neighborhoods within two matrices.
     *
     * @param handle A resource handle for creating internal matrices.
     * @param nRad Neighborhood radius; the distance from the center of a
     * neighborhood to its edge.
     * @param height The height of expected matrices. That is, matrices that
     * will be passed to the set method.
     * @param width The width of expected matrices.
     
     */
    public NeighborhoodProductSums(Handle handle, int nRad, int height, int width) {
        this.nRad = nRad;
        this.height = height;
        this.width = width;
        inRowSum = new Matrix(handle, height, width);
        ebeStorage = new Matrix(handle, height, width);
        halfNOnes = new Vector(handle, nRad + 1).fill(1);
    }

    /**
     * Computes neighborhood element-wise multiplication of matrices a and b.
     * Divided into row and column stages for better performance. Then places in
     * result the summation of all the ebe products in the neighborhood of an
     * index pair in that index pair (column major order).
     *
     * @param a The first matrix.
     * @param b The second matrix.
     * @param result Store the result here in column major order. Note that the
     * increment of this vector is probably not one. Be sure this is set to 0's
     * before passing it.
     *
     */
    public void set(Matrix a, Matrix b, Vector result) {

        ebeStorage.asVector().ebeMultiplyAndSet(a.asVector(), b.asVector());

        inRowSumsEdge();
        inRowSumNearEdge();
        inRowSumCenter();

        VectorsStride resultRows = result.subVectors(1, height, width, a.colDist);
        
        nSumEdge(resultRows);
        nSumNearEdge(resultRows);
        nSumCenter(resultRows);
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
        inRowSum.getColumn(0).multiplyAndSet(
                ebeStorage.getSubMatrixCols(0, nRad + 1),
                halfNOnes
        );
        inRowSum.getColumn(width - 1).multiplyAndSet(
                ebeStorage.getSubMatrixCols(width - nRad - 1, width),
                halfNOnes
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
            inRowSum.getColumn(i).addAndSet(
                    1, ebeStorage.getColumn(i + nRad),
                    1, inRowSum.getColumn(i - 1)
            );
            int colInd = width - 1 - i;
            inRowSum.getColumn(colInd).addAndSet(
                    1, ebeStorage.getColumn(colInd - nRad),
                    1, inRowSum.getColumn(colInd + 1)
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
            Vector column = inRowSum.getColumn(colIndex);
            column.addAndSet(
                    -1, ebeStorage.getColumn(colIndex - nRad - 1),
                    1, ebeStorage.getColumn(colIndex + nRad)
            );
            column.addToMe(1, inRowSum.getColumn(colIndex - 1));
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
    private void nSumEdge(VectorsStride resultRows) {
        resultRows.getVector(0).multiplyAndSet(
                halfNOnes,
                inRowSum.getSubMatrixRows(0, nRad + 1)
        );
        resultRows.getVector(height - 1).multiplyAndSet(
                halfNOnes,
                inRowSum.getSubMatrixRows(height - nRad - 1, height)
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
    private void nSumNearEdge(VectorsStride resultRows) {
        for (int i = 1; i < nRad + 1; i++) {
            int rowInd = i;
            Vector nSumRow = resultRows.getVector(rowInd);
            nSumRow.addToMe(1, inRowSum.getRow(rowInd + nRad));
            nSumRow.addToMe(1, resultRows.getVector(rowInd - 1));

            rowInd = height - 1 - i;
            nSumRow = resultRows.getVector(rowInd);
            nSumRow.addToMe(1, inRowSum.getRow(rowInd - nRad));
            nSumRow.addToMe(1, resultRows.getVector(rowInd + 1));

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
    private void nSumCenter(VectorsStride resultRows) {
        for (int rowIndex = nRad + 1; rowIndex < height - nRad; rowIndex++) {
            Vector nSumsRow = resultRows.getVector(rowIndex);
            nSumsRow.addToMe(-1, inRowSum.getRow(rowIndex - nRad - 1));
            nSumsRow.addToMe(1, inRowSum.getRow(rowIndex + nRad));

            nSumsRow.addToMe(1, resultRows.getVector(rowIndex - 1));
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
