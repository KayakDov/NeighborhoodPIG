package JCudaWrapper.algebra;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.DArray3d;

/**
 * Any column major matrix like think could implement this class.
 * @author E. Dov Neimand
 */
public interface ColumnMajor {
    
    public int colDist();
    
    /**
     * Returns the column-major vector index of the given row and column.
     *
     * @param row The row index.
     * @param col The column index.
     * @return The vector index: {@code col * colDist + row}.
     */
    default public int index(int row, int col){
        return col*colDist() + row;
    }
    
    /**
     * Gets the underlying array for this data structure.
     * @return The GPU array.
     */
    public Array array();
    
    
}
