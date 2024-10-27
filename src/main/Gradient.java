package main;



import algebra.MatricesStride;
import algebra.Matrix;
import algebra.Vector;
import algebra.VectorsStride;
import array.DArray;
import resourceManagement.Handle;


/**
 * The gradient for each pixel.
 * @author E. Dov Neimand
 */
public class Gradient {
    
    private Matrix dX, dY;
    
    /**
     * Computes the gradients of an image in both the x and y directions.
     * Gradients are computed using central differences for interior points and
     * forward/backward differences for boundary points.
     *
     * @param pic The pixel intensity values matrix.
     * @param hand Handle to manage GPU memory or any other resources.
     
     */
    public Gradient(Matrix pic, Handle hand) {
        int width = pic.getWidth(),  height = pic.getHeight();

        dX = new Matrix(hand, height, width);
        dY = new Matrix(hand, height, width);

        computeBoundaryGradients(pic, dX, dY, height, width);
        computeInteriorGradients(hand, pic, dX, dY, width, height);
    }


    /**
     * Computes the gradients at the boundary of the image using forward and
     * backward differences. This method handles the first and last rows and
     * columns of the image, where gradients are calculated with one-sided
     * differences.
     *
     * @param pic The input matrix containing pixel intensity values.
     * @param dX The matrix to store gradients along the x-axis.
     * @param dY The matrix to store gradients along the y-axis.
     * @param hand The handle object for managing resources and operations.
     * @param height The height of the image (number of rows).
     * @param width The width of the image (number of columns).
     */
    private void computeBoundaryGradients(Matrix pic, Matrix dX, Matrix dY, int height, int width) {
        
            Matrix col0 =  pic.getColumnMatrix(0);
            Matrix colLast = pic.getColumnMatrix(width - 1);
            
            dX.getColumnMatrix(0).addAndSet(-1, col0, 1, pic.getColumnMatrix(1));
            dX.getColumnMatrix(width - 1).addAndSet(-1, pic.getColumnMatrix(width - 2), 1, colLast);
            dX.getColumnMatrix(1).addAndSet(-0.5, col0, 0.5, pic.getColumnMatrix(2));
            dX.getColumnMatrix(width - 2).addAndSet(-0.5, pic.getColumnMatrix(width - 3), 0.5, colLast);
            
            Matrix row0 =  pic.getRowMatrix(0), rowLast = pic.getRowMatrix(height - 1);
            
            dY.getRowMatrix(0).addAndSet(-1, row0, 1, pic.getRowMatrix(1));
            dY.getRowMatrix(height - 1).addAndSet(-1, pic.getRowMatrix(height - 2), 1, rowLast);
            dY.getRowMatrix(1).addAndSet(-0.5, row0, 0.5, pic.getRowMatrix(2));
            dX.getRowMatrix(height - 2).addAndSet(-0.5, pic.getRowMatrix(height - 3), 0.5, rowLast);
    }


    /**
     * Computes the gradients for the interior pixels using higher-order
     * differences. This method calculates the gradients for pixels that are not
     * near the boundary, using a higher-order finite difference scheme for
     * increased accuracy.
     *
     * @param pic The input matrix containing pixel intensity values.
     * @param dX The matrix to store gradients along the x-axis.
     * @param dY The matrix to store gradients along the y-axis.
     * @param hand The handle object for managing resources and operations.
     * @param width The width of the image (number of columns).
     * @param height The height of the image (number of rows).
     */
    private void computeInteriorGradients(Handle hand, Matrix pic, Matrix dX, Matrix dY, int width, int height) {
        try (VectorsStride diff = new VectorsStride(hand, new DArray(hand, -1.0 / 12, 2.0 / 3, 0, -2.0 / 3, 1.0 / 12).getStrided(0, dX.getWidth() - 4, 5),1)) {
            // Interior x gradients (third column to second-to-last)
            VectorsStride dXColumns = dX.columns();
            MatricesStride columnBlocks = new MatricesStride(
                    hand, 
                    pic.dArray().getStrided(height, width - 4, height*diff.getSubVecDim()), 
                    height, 
                    diff.getSubVecDim(),
                    height
            );
            
            dXColumns.setMatVecMult(columnBlocks, diff);

            // Interior y gradients (third row to second-to-last)
            VectorsStride dYRows = dY.rows();
            MatricesStride rowBlocks = new MatricesStride(
                    hand, 
                    pic.dArray().getStrided(1, height - 4, height*(width-1)), 
                    diff.getSubVecDim(), 
                    width,
                    height
            );
            dYRows.setVecMatMult(diff, rowBlocks);
        }
    }

    /**
     * An unmodifiable x gradient matrix.
     * @return An unmodifiable x gradient matrix.
     */
    public Matrix getdX() {
        return dX.unmodifable();
    }

    /**
     * An unmodifiable y gradient matrix.
     * @return An unmodifiable y gradient matrix.
     */
    public Matrix getdY() {
        return dY.unmodifable();
    }
    
    
    
    
}
