
package JCudaWrapper.algebra;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.resourceManagement.Handle;

/**
 * A 3d matrix class.
 * @author E. Dov Neimand
 */
public class TensorOrd3 extends TensorOrd3StrideDim {
    
    private final DArray3d data;
    
    /**
     * Constructor for a 3d matrix
     * @param handle  The handle.
     * @param data The underlying data.
     */
    public TensorOrd3(Handle handle, DArray3d data) {
        super(handle, data.entriesPerLine(), data.linesPerLayer(), data.linesPerLayer(), 1);
        this.data = data;
    }
    
    /**
     * Constructor for a 3d matrix
     * @param handle  The handle.
     * @param height The height of the matrix (number of columns).
     * @param width The width of the matrix (number of rows).
     * @param depth Thee depth of the matrix (number of layers).
     */
    public TensorOrd3(Handle handle, int height, int width, int depth) {
        super(handle, height, width, depth,1);
        data = new DArray3d(height, width, depth);
    }

    /**
     * The distance between the first element of each layer.
     * @return The distance between the first element of each layer.
     */
    public int layerDist(){
        return data.ld()*width;
    }
    
    /**
     * The column major index of the desired element.
     * @param row The row of the desired element.
     * @param col The column of the desired element.
     * @param layer The layer of the desired element.
     * @return The column major index of the desired element.
     */
    private int index(int row, int col, int layer){
        return layer*layerDist() + col*colDist() + row;
    }
    
    /**
     * A sub Matrix.
     * @param startRow The startinf row of the submatrix inclusive.
     * @param subHeight The end row of the submatrix exclusive.
     * @param startCol The starting column of the submatrix inclusive.
     * @param subWidth The width of the sub tensor
     * @param startLayer The starting layer of the matrix inlcusive.
     * @param subDepth
     * @return The requested submatrix.
     */
    public TensorOrd3 subMatrix(int startRow, int subHeight, int startCol, int subWidth, int startLayer, int subDepth){
        return new TensorOrd3(handle, data.sub(startRow, subHeight, startCol, subWidth, startLayer, subDepth));
    }
    
    /**
     * Adds the product of the row vector and a matrix to this.
     * @param scalar Times the matrix to be added,
     * @param vec A vector representing a row.
     * @param mat A matrix.
     * @param timesThis Times this before anything is added.
     * @return this.
     */
    public TensorOrd3 addProductCol(double scalar, Vector vec, TensorOrd3 mat, double timesThis){
        throw new UnsupportedOperationException();
    }
        
    /**
     * Adds the product of a matrix and a column vector to this.
     * @param scalar Times the matrix to be added,
     * @param mat A matrix.
     * @param timesThis Times this before anything is added.
     * @param vec A vector representing a row.
     * @return this.
     */
    public TensorOrd3 addProductRow(double scalar, TensorOrd3 mat, Vector vec, double timesThis){
        throw new UnsupportedOperationException();
    }
    
    /**
     * Adds the product of a matrix and a depth vector to this.
     * @param scalar Times the matrix to be added,
     * @param timesThis Times this before anything is added.
     * @param mat A matrix.
     * @param vec A vector representing a depth vector.
     * @return this.
     */
    public TensorOrd3 addProductDepth(double scalar, TensorOrd3 mat, Vector vec, double timesThis){
        throw new UnsupportedOperationException();
    }
    
    /**
     * The desired row in every layer.
     * @param row the desired row.
     * @return A layer of this matrix.
     */
    public Matrix getRows(int row){
        throw new UnsupportedOperationException();
    }
    
    /**
     * The desired column in every layer.
     * @param col The index of the column.
     * @return The desired column in every layer.
     */
    public Matrix getColumns(int col){
        throw new UnsupportedOperationException();
    }
    
    /**
     * A layer of this matrix.
     * @param layer The index of the layer.
     * @return The desired layer of this matrix.
     */
    public Matrix getLayer(int layer){
        throw new UnsupportedOperationException();
    }
    
    /**
     * Sets this tensor to be the sum alpha*a + beta*b.
     * @param alpha A constant times a.
     * @param a The first matrix to be added.
     * @param beta A constant times b.
     * @param b The second matrix to be added.
     * @return this.
     */
    public TensorOrd3 setSum(double alpha, TensorOrd3 a, double beta, TensorOrd3 b){
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray3d array() {
        return data;
    }

    
    /**
     * {@inheritDoc }
     */
    @Override
    public void close() throws Exception {
        data.close();
    }

    @Override
    public int colDist() {
        return data.ld();
    }
    
}
