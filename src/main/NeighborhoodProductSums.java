package main;

import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.Vector;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
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

//    private final Vector halfNOnes;
    private final Matrix ebeStorage, sumLocalRowElements;
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
     *
     */
    public NeighborhoodProductSums(Handle handle, int nRad, int height, int width) {
        this.nRad = nRad;
        this.height = height;
        this.width = width;
        sumLocalRowElements = new Matrix(handle, height, width);
        ebeStorage = new Matrix(handle, height, width);
//        halfNOnes = new Vector(handle, nRad + 1).fill(1);
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
     * increment of this vector is probably not one.
     */
    public void set(Matrix a, Matrix b, Vector result) {

        Handle hand = a.getHandle();

        new Vector(hand, ebeStorage.dArray(), 1)
                .ebeSetProduct(
                        new Vector(hand, a.dArray(), 1),
                        new Vector(hand, b.dArray(), 1)
                );

        KernelManager.get("neighborhoodSum").map(hand,
                ebeStorage.dArray(),
                width,
                sumLocalRowElements.dArray(),
                1,
                height,
                IArray.cpuTrue(),
                IArray.cpuPointer(nRad)
        );

        KernelManager.get("neighborhoodSum").map(hand,
                sumLocalRowElements.dArray(),
                height,
                result.dArray(),
                result.inc(),
                width,
                IArray.cpuFalse(),
                IArray.cpuPointer(nRad)
        );
//       Uses library methods instead of kernel.  The library methods are thought to be slower on account of repeated calls to the gpu. 
//        sumLocalRowElementsEdge();
//        sumLocalRowElementsNearEdge();
//        sumLocalRowElementsCenter();
//        
//        VectorsStride resultRows = result.subVectors(1, width, a.colDist, height);
//
//        nSumEdge(resultRows);
//        nSumNearEdge(resultRows);
//        nSumCenter(resultRows);       
    }

//    /**
//     * Handles column summation for the first and last columns.
//     *
//     * @param inRowSum Matrix to store intermediate row sums.
//     * @param ebeStorage Element-wise multiplied matrix.
//     * @param nRad Neighborhood radius.
//     * @param height Height of the matrix.
//     * @param width Width of the matrix.
//     * @param halfNOnes Vector of ones used for summing the first and last
//     * columns.
//     */
//    private void sumLocalRowElementsEdge() {
//        
//        sumLocalRowElements.getColumn(0).setProduct(
//                ebeStorage.getColumns(0, nRad + 1),
//                halfNOnes
//        );
//        sumLocalRowElements.getColumn(width - 1).setProduct(
//                ebeStorage.getColumns(width - nRad - 1, width),
//                halfNOnes
//        );        
//    }
//
//    /**
//     * Handles column summation for columns near the first and last.
//     *
//     * @param inRowSum Matrix to store intermediate row sums.
//     * @param ebeStorage Element-wise multiplied matrix.
//     * @param nRad Neighborhood radius.
//     * @param width Width of the matrix.
//     */
//    private void sumLocalRowElementsNearEdge() {
//        for (int i = 1; i < nRad + 1; i++) {
//            sumLocalRowElements.getColumn(i).setSum(
//                    1, ebeStorage.getColumn(i + nRad),
//                    1, sumLocalRowElements.getColumn(i - 1)
//            );
//            int colInd = width - 1 - i;
//            sumLocalRowElements.getColumn(colInd).setSum(
//                    1, ebeStorage.getColumn(colInd - nRad),
//                    1, sumLocalRowElements.getColumn(colInd + 1)
//            );
//        }
//    }
//
//    /**
//     * Handles column summation for the central columns (Stage III).
//     *
//     * @param inRowSum Matrix to store intermediate row sums.
//     * @param ebeStorage Element-wise multiplied matrix.
//     * @param nRad Neighborhood radius.
//     * @param width Width of the matrix.
//     */
//    private void sumLocalRowElementsCenter() {
//        for (int colIndex = nRad + 1; colIndex + nRad < width; colIndex++)
//            sumLocalRowElements.getColumn(colIndex).setSum(
//                    -1, ebeStorage.getColumn(colIndex - nRad - 1),
//                    1, ebeStorage.getColumn(colIndex + nRad)
//            ).add(1, sumLocalRowElements.getColumn(colIndex - 1));
//    }
//
//    /**
//     * Handles row summation for the first and last rows (Stage I).
//     *
//     * @param nSums Matrix on the result vector to store the results.
//     * @param inRowSum Matrix containing intermediate row sums.
//     * @param nRad Neighborhood radius.
//     * @param width Width of the matrix.
//     * @param halfNOnes Vector of ones used for summing the first and last rows.
//     */
//    private void nSumEdge(VectorsStride resultRows) {
//        resultRows.getVector(0).setProduct(
//                halfNOnes,
//                sumLocalRowElements.getRows(0, nRad + 1)
//        );
//        resultRows.getVector(height - 1).setProduct(
//                halfNOnes,
//                sumLocalRowElements.getRows(height - nRad - 1, height)
//        );
//    }
//
//    /**
//     * Handles row summation for rows near the first and last (Stage II).
//     *
//     * @param nSums Matrix on the result vector to store the results.
//     * @param inRowSum Matrix containing intermediate row sums.
//     * @param nRad Neighborhood radius.
//     * @param height Height of the matrix.
//     */
//    private void nSumNearEdge(VectorsStride resultRows) {
//        for (int i = 1; i < nRad + 1; i++) {
//
//            int rowInd = i;
//            resultRows.getVector(rowInd).setSum(
//                    1, sumLocalRowElements.getRow(rowInd + nRad),
//                    1, resultRows.getVector(rowInd - 1)
//            );
//
//            rowInd = height - 1 - i;
//            resultRows.getVector(rowInd).setSum(
//                    1, sumLocalRowElements.getRow(rowInd - nRad),
//                    1, resultRows.getVector(rowInd + 1)
//            );
//        }
//    }
//
//    /**
//     * Handles row summation for the central rows.
//     *
//     * @param nSums Matrix on the result vector to store the results.
//     * @param inRowSum Matrix containing intermediate row sums.
//     * @param nRad Neighborhood radius.
//     * @param height Height of the matrix.
//     */
//    private void nSumCenter(VectorsStride resultRows) {
//        for (int rowIndex = nRad + 1; rowIndex < height - nRad; rowIndex++) {
//
//            resultRows.getVector(rowIndex).setSum(
//                    -1, sumLocalRowElements.getRow(rowIndex - nRad - 1),
//                    1, sumLocalRowElements.getRow(rowIndex + nRad)
//            ).add(1, resultRows.getVector(rowIndex - 1));
//        }
//    }
    /**
     * Cleans up allocated memory on the gpu.
     */
    @Override
    public void close() {
//        halfNOnes.close();
        ebeStorage.close();
        sumLocalRowElements.close();
    }
}
