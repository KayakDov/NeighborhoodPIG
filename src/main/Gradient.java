package main;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.resourceManagement.Handle;
import java.util.function.IntFunction;

/**
 * The gradient for each pixel.
 *
 * @author E. Dov Neimand
 */
public class Gradient implements AutoCloseable{

    private Matrix dX, dY;

    /**
     * Computes the gradients of an image in both the x and y directions.
     * Gradients are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     *
     */
    public Gradient(Matrix pic, Handle hand) {
        int width = pic.getWidth(), height = pic.getHeight();

        dX = new Matrix(hand, height, width);
        dY = new Matrix(hand, height, width);

        computeBoundaryGradients(i -> pic.getColumn(i), i -> dX.getColumn(i), width);
        computeBoundaryGradients(i -> pic.getRow(i), i -> dY.getRow(i), height);

        computeInteriorGradients(hand, pic, width, height, height, diff.length, dX.columns());
        computeInteriorGradients(hand, pic, height, 1, diff.length, width, dY.rows());
    }

    /**
     * Sets the border rows (columns) of the gradient matrices.
     *
     * Computes the gradients at the boundary of the image using forward and
     * backward differences. This method handles the first and last rows and
     * columns of the image, where gradients are calculated with one-sided
     * differences.
     *
     * The matrices from which pic and dM are taken should be pixel intensities.
     *
     * @param pic The picture the gradient is being taken from. The output of
     * this method should be the rows (columns) that the gradient is taken over.
     * @param dM Where the gradient is to be stored. The output should be the
     * rows (columns) where the gradient is stored. Use rows (columns) for dY
     * (dX).
     * @param length The max row (column) index exclusive.
     */
    private void computeBoundaryGradients(IntFunction<Matrix> pic, IntFunction<Matrix> dM, int length) {
        dM.apply(0).addAndSet(-1, pic.apply(0), 1, pic.apply(1));
        dM.apply(length - 1).addAndSet(-1, pic.apply(length - 2), 1, pic.apply(length - 1));
        dM.apply(1).addAndSet(-0.5, pic.apply(0), 0.5, pic.apply(2));
        dM.apply(length - 2).addAndSet(-0.5, pic.apply(length - 3), 0.5, pic.apply(length - 1));
    }

    /**
     * An array used for differentiation.
     */
    private static final DArray diff;

    static {
        try (Handle hand = new Handle()) {
            diff = new DArray(hand, 1.0 / 12, -2.0 / 3, 0, 2.0 / 3, -1.0 / 12);
        }
    }

    /**
     * Computes the gradients for the interior pixels using higher-order
     * differences. This method calculates the gradients for pixels that are not
     * near the boundary, using a higher-order finite difference scheme for
     * increased accuracy.
     *
     * The blocks are sub matrices of rows or columns that are added/subtracted
     * according to diff to find the gradient.
     *
     * @param hand The handle.
     * @param pic The picture over which the gradient is taken.
     * @param length Either the height or the width as appropriate.
     * @param blockStride This should be one if the blocks are made of rows
     * @param blockHeight The height of each block.  For row blocks this should be diff.length and for col blocks this should be height.
     * @param blockWidth see block height but opposite.
     * @param target Where the results are stored.  This should either be dX.columns() or dY.rows()
     */
    private void computeInteriorGradients(Handle hand, Matrix pic, int length, int blockStride, int blockHeight, int blockWidth, VectorsStride target) {

        // Interior x gradients (third column to second-to-last)
        int numBlocks = length - diff.length + 1;

        VectorsStride diffVecs = new VectorsStride(hand, diff, 1, diff.length, 0, numBlocks);

        MatricesStride blocks = new MatricesStride(hand, pic.dArray(), blockHeight, blockWidth, pic.colDist, blockStride, numBlocks);
        
        target = target.subBatch(2, numBlocks);

        if (blocks.height == diff.length) target.setVecMatMult(diffVecs, blocks);
        else target.setMatVecMult(blocks, diffVecs);
    }

    /**
     * An unmodifiable x gradient matrix.
     *
     * @return An unmodifiable x gradient matrix.
     */
    public Matrix x() {
        return dX;
    }

    /**
     * An unmodifiable y gradient matrix.
     *
     * @return An unmodifiable y gradient matrix.
     */
    public Matrix y() {
        return dY;
    }

    @Override
    public void close() {
        dX.close();
        dY.close();
    }

}
