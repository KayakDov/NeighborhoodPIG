package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.resourceManagement.Handle;
import java.awt.Dimension;
import java.util.Arrays;
import org.apache.commons.math3.exception.*;
import org.apache.commons.math3.linear.*;
import JCudaWrapper.array.DSingleton;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;

/**
 * Represents a matrix stored on the GPU. For more information on jcuda
 *
 * TODO: implement extensions of this class to include banded, symmetric banded,
 * triangular packed, symmetric matrices.
 *
 * http://www.jcuda.org/jcuda/jcublas/JCublas.html
 */
public class Matrix implements AutoCloseable, ColumnMajor {

    /**
     * The number of rows in the matrix.
     */
    private final int height;

    /**
     * The number of columns in the matrix.
     */
    private final int width;

    /**
     * The distance between the first element of each column in memory.
     * <p>
     * Typically, this is equal to the matrix height, but if this matrix is a
     * submatrix, `colDist` may differ, indicating that the matrix data is
     * stored with non-contiguous elements in memory.
     * </p>
     */
    public final int colDist;

    /**
     * The underlying GPU data storage for this matrix.
     */
    protected final DArray data;

    /**
     * Handle for managing JCublas operations, usually unique per thread.
     */
    protected Handle handle;

    /**
     * Constructs a new Matrix from a 2D array, where each inner array
     * represents a column of the matrix.
     *
     * @param handle The handle for JCublas operations, required for matrix
     * operations on the GPU.
     * @param matrix A 2D array, where each sub-array is a column of the matrix.
     */
    public Matrix(Handle handle, double[][] matrix) {
        this(handle, matrix[0].length, matrix.length);
        Matrix.this.set(0, 0, matrix);
    }

    /**
     * Constructs a Matrix from a single array representing a column-major
     * matrix.
     *
     * @param array The array storing the matrix in column-major order.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(Handle handle, DArray array, int height, int width) {
        this(handle, array, height, width, height);
    }

    /**
     * Constructs a new Matrix from an existing RealMatrix object, copying its
     * data to GPU memory.
     *
     * @param mat The matrix to be copied to GPU memory.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(Handle handle, RealMatrix mat) {
        this(handle, mat.getData());
    }

    /**
     * Creates a shallow copy of an existing Matrix, referencing the same data
     * on the GPU without copying. Changes to this matrix will affect the
     * original and vice versa.
     *
     * @param mat The matrix to create a shallow copy of.
     */
    public Matrix(Matrix mat) {
        this(mat.handle, mat.data, mat.height, mat.width, mat.colDist);
    }

    /**
     * Constructs a new Matrix from an existing data pointer on the GPU.
     *
     * @param array Pointer to the data on the GPU.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param distBetweenFirstElementOfColumns The distance between the first
     * element of each column in memory, usually equal to height. If this is a
     * submatrix, it may differ.
     * @param handle The handle for GPU operations.
     */
    public Matrix(Handle handle, DArray array, int height, int width, int distBetweenFirstElementOfColumns) {
        this.height = height;
        this.width = width;
        this.data = array;
        this.handle = handle;
        this.colDist = distBetweenFirstElementOfColumns;
    }

    /**
     * Constructs an empty matrix of specified height and width.
     *
     * @param handle The handle for GPU operations.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     */
    public Matrix(Handle handle, int height, int width) {
        this(handle, DArray.empty(height * width), height, width);
    }

    /**
     * Returns the height (number of rows) of the matrix.
     *
     * @return The number of rows in the matrix.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the width (number of columns) of the matrix.
     *
     * @return The number of columns in the matrix.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Multiplies two matrices, adding the result into this matrix. The result
     * is inserted into this matrix as a submatrix.
     *
     * @param transposeA True if the first matrix should be transposed.
     * @param transposeB True if the second matrix should be transposed.
     * @param timesAB Scalar multiplier for the product of the two matrices.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param timesThis Scalar multiplier for the elements in this matrix.
     * @return This matrix after the operation.
     */
    public Matrix addProduct(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {

        Dimension aDim = new Dimension(transposeA ? a.height : a.width, transposeA ? a.width : a.height);
        Dimension bDim = new Dimension(transposeB ? b.height : b.width, transposeB ? b.width : b.height);
        Dimension result = new Dimension(bDim.width, aDim.height);

        checkRowCol(result.height - 1, result.width - 1);

        data.addProduct(handle, transposeA, transposeB,
                aDim.height, bDim.width, aDim.width, timesAB,
                a.data, a.colDist, b.data, b.colDist,
                timesThis, colDist);
        return this;
    }

    /**
     * @see Matrix#addProduct(processSupport.Handle, boolean, boolean, double,
     * algebra.Matrix, algebra.Matrix, double) timesThis is set to 0, transpose
     * values are false, and timesAB is 1.
     * @param a To be multiplied by the first matrix.
     * @param b To be multiplied by the second matrix.
     * @return This matrix.
     */
    public Matrix setToProduct(Matrix a, Matrix b) {
        return addProduct(false, false, 1, a, b, 0);
    }

    /**
     * @see Matrix#addProduct(processSupport.Handle, boolean, boolean, double,
     * algebra.Matrix, algebra.Matrix, double) Uses the default handle.
     * @param transposeA True to transpose A, false otherwise.
     * @param transposeB True to transpose B, false otherwise.
     * @param timesAB TO me multiplied by AB.
     * @param a The A matrix.
     * @param b The B matrix.
     * @param timesThis To be multiplied by this.
     * @return this.
     */
    public Matrix setToProduct(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {
        return addProduct(transposeA, transposeB, timesAB, a, b, timesThis);
    }

    /**
     * Returns the row index corresponding to the given column-major vector
     * index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The row index.
     */
    private int rowIndex(int vectorIndex) {
        return vectorIndex % height;
    }

    /**
     * Returns the column index corresponding to the given column-major vector
     * index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The column index.
     */
    private int columnIndex(int vectorIndex) {
        return vectorIndex / height;
    }

    /**
     * Copies a matrix to the GPU and stores it in the internal data structure.
     *
     * @param toRow The starting row index in this matrix.
     * @param toCol The starting column index in this matrix.
     * @param matrix The matrix to be copied, represented as an array of
     * columns.
     */
    private final void set(int toRow, int toCol, double[][] matrix) {
        for (int col = 0; col < Math.min(width, matrix.length); col++) {
            data.set(handle, matrix[col], index(toRow, toCol + col));
        }
    }

    /**
     * Performs matrix addition or subtraction.
     *
     * <p>
     * This function computes this = alpha * A + beta * B, where A and B are
     * matrices.
     * </p>
     *
     * @param handle The handle.
     * @param transA specifies whether matrix A is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param transB specifies whether matrix B is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @return this
     *
     */
    public Matrix setSum(Handle handle, boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {

        if (transA) checkRowCol(a.width - 1, a.height - 1);
        else checkRowCol(a.height - 1, a.width - 1);
        if (transB) checkRowCol(b.width - 1, b.height - 1);
        else  checkRowCol(b.height - 1, b.width - 1);
        

        data.setSum(handle, transA, transB, height, width, alpha, a.data, a.colDist, beta, b.data, b.colDist, colDist);

        return this;
    }

    /**
     * @see Matrix#setSum(boolean, boolean, double, algebra.Matrix, double, algebra.Matrix) Uses default handle.
     * @param transA True to transpose A.
     * @param transB True to transpose B.
     * @param alpha the multiply by A.
     * @param a The A matrix.
     * @param beta To multiply by B.
     * @param b The B matrix.
     * @return This.
     */
    public Matrix setSum(boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {
        return setSum(handle, transA, transB, alpha, a, beta, b);
    }

    /**
     * @see Matrix#setSum(boolean, boolean, double, algebra.Matrix, double, algebra.Matrix) Uses default handle.
     * @param alpha the multiply by A.
     * @param a The A matrix.
     * @param beta To multiply by B.
     * @param b The B matrix.
     * @return This.
     */
    public Matrix setSum(double alpha, Matrix a, double beta, Matrix b) {
        return setSum(handle, false, false, alpha, a, beta, b);
    }

    /**
     * Multiplies everything in this matrix by a scalar
     *
     * @param d The scalar that does the multiplying.
     * @return A new matrix equal to this matrix times a scalar.
     */
    public Matrix multiply(double d) {
        return setSum(d, this, 0, this);
    }

    /**
     * Fills this matrix with @code{d}, overwriting whatever is there.
     *
     * @param scalar The value to fill the matrix with.
     * @return this.
     */
    public Matrix fill(double scalar) {
        data.fillMatrix(handle, height, width, colDist, scalar);
        return this;
    }

    /**
     * Adds a scalar to every element of this matrix.
     *
     * @param d The scalar to be added.
     * @return this.
     */
    public Matrix add(double d) {
        try (DSingleton sing = new DSingleton(handle, d)) {
            KernelManager.get("addScalarToMatrix").map(handle, sing, colDist, data, height, size());
            return this;
        }
    }

    /**
     * Inserts anther matrix into this matrix at the given index.
     *
     * @param handle The handle with which to perform this operation.
     * @param other The matrix to be inserted
     * @param row the row in this matrix where the first row of the other matrix
     * is inserted.
     * @param col The column in this matrix where the first row of the other
     * matrix is inserted.
     * @return this.
     *
     */
    public Matrix insert(Handle handle, Matrix other, int row, int col) {
        checkSubMatrixParameters(row, row + other.height, col, col + other.width);

        getSubMatrix(row, row + other.height, col, col + other.width)
                .setSum(1, other, 0, other);

        return this;
    }

    /**
     * @see Matrix#insert(processSupport.Handle, algebra.Matrix, int, int)
     * except with default handle.
     *
     * @param other
     * @param row
     * @param col
     * @return
     */
    public Matrix insert(Matrix other, int row, int col) {
        return insert(handle, other, row, col);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return rows().toString();

    }

    /**
     * Gets the entry at the given row and column.
     *
     * @param row The row of the desired entry.
     * @param column The column of the desired entry.
     * @return The entry at the given row and column.
     */
    public double get(int row, int column) {
        return data.get(index(row, column)).getVal(handle);

    }

    /**
     * The dimensions of a submatrix.
     *
     * @param startRow The top row of the submatrix.
     * @param endRow The bottom row of the submatrix, exclusive.
     * @param startColumn The first column of a submatrix.
     * @param endColumn The last column of the submatrix, exclusive.
     * @return The dimensions of a submatrix.
     */
    private Dimension subMatrixDimensions(int startRow, int endRow, int startColumn, int endColumn) {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);
        return new Dimension(endColumn - startColumn, endRow - startRow);
    }

    /**
     * Does some basic checks on the validity of the subMatrix parameters. Throw
     * exceptions if there are any problems.
     *
     * @param startRow inclusive
     * @param endRow exclusive
     * @param startColumn inclusive
     * @param endColumn exclusive
     * @throws OutOfRangeException
     * @throws NumberIsTooSmallException
     */
    private void checkSubMatrixParameters(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkRowCol(endRow - 1, endColumn - 1);
        checkRowCol(startRow, startColumn);
        if (startColumn > endColumn) {
            throw new NumberIsTooSmallException(endColumn, startColumn, true);
        }
        if (startRow > endRow) {
            throw new NumberIsTooSmallException(endRow, startRow, true);
        }

    }

    /**
     * Passes by reference. Changes to the sub matrix will effect the original
     * matrix and vice versa.
     *
     * @param startRow The starting row of the submatrix.
     * @param endRow The end row of the submatrix exclusive.
     * @param startColumn The starting column of the submatrix.
     * @param endColumn The end column exclusive.
     * @return The submatrix.
     * @throws OutOfRangeException
     * @throws NumberIsTooSmallException
     */
    public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        return new Matrix(
                handle,
                data.subArray(index(startRow, startColumn), (dim.width - 1) * colDist + dim.height),
                dim.height,
                dim.width,
                colDist);
    }

    /**
     * A submatrix consisting of the given rows.
     *
     * @param startRow The first row.
     * @param endRow The last row, exclusive.
     * @return A submatrix from the given rows.
     */
    public Matrix getRows(int startRow, int endRow) {
        return getSubMatrix(startRow, endRow, 0, getWidth());
    }

    /**
     * A submatrix consisting of the given columns.
     *
     * @param startCol The first column.
     * @param endCol The last column, exclusive.
     * @return A submatrix from the given rows.
     */
    public Matrix getColumns(int startCol, int endCol) {
        return getSubMatrix(0, getHeight(), startCol, endCol);
    }

    /**
     * If the row is outside of this matrix, an exception is thrown.
     *
     * @param row The row to be checked.
     * @throws OutOfRangeException
     */
    private void checkRow(int row) throws OutOfRangeException {
        if (row < 0 || row >= height) {
            throw new OutOfRangeException(row, 0, height);
        }
    }

    /**
     * If the column is outside of this matrix, an exception is thrown.
     *
     * @param col The column to be checked.
     * @throws OutOfRangeException
     */
    private void checkCol(int col) {
        if (col < 0 || col >= width) {
            throw new OutOfRangeException(col, 0, width);
        }
    }

    /**
     * If either the row or column are out of range, an exception is thrown.
     *
     * @param row The row to be checked.
     * @param col The column to be checked.
     */
    private void checkRowCol(int row, int col) throws OutOfRangeException {
        checkRow(row);
        checkCol(col);
    }

    /**
     * Checks if any of the objects passed are null, and if they are, throws a
     * null argument exception.
     *
     * @param o To be checked for null values.
     */
    private void checkForNull(Object... o) {
        if (Arrays.stream(o).anyMatch(obj -> obj == null)) {
            throw new NullArgumentException();
        }
    }

    /**
     * The number of elements in this matrix.
     *
     * @return The number of elements in this matrix.
     */
    public int size() {
        return width * height;
    }

    /**
     * Checks if the two methods are equal to within an epsilon margin of error.
     *
     * @param other A matrix that might be equal to this one.
     * @param epsilon The acceptable margin of error.
     * @param workSpace Should be the size of the matrix.
     * @return True if the matrices are very close to one another, false
     * otherwise.
     */
    public boolean equals(Matrix other, double epsilon, DArray workSpace) {
        if (height != other.height || width != other.width) return false;

        return new Matrix(handle, data, height, width).setSum(1, this, -1, other)
                .frobeniusNorm(workSpace.subArray(0, width)) <= epsilon;
    }

    /**
     * A copy of this matrix.
     *
     * @return A copy of this matrix.
     */
    public Matrix copy() {
        if (height == colDist) {
            return new Matrix(handle, data.copy(handle), height, width);
        }

        Matrix copy = new Matrix(handle, height, width);

        copy.setSum(1, this, 0, this);

        return copy;
    }

    /**
     * Sets an entry.
     *
     * @param row The row of the entry.
     * @param column The column of the entry.
     * @param value The value to be placed at the entry.
     */
    public void set(int row, int column, double value) {
        data.set(handle, index(row, column), value);
    }

    /**
     * gets the row.
     *
     * @param row the index of the desired row.
     * @return The row at the requested index.
     * @throws OutOfRangeException
     */
    public Vector getRow(int row) throws OutOfRangeException {
        return new Vector(handle, data.subArray(row), colDist);
    }

    /**
     * Gets the requested column.
     *
     * @param column The index of the desired column.
     * @return The column at the submited index.
     * @throws OutOfRangeException
     */
    public Vector getColumn(int column) throws OutOfRangeException {
        return new Vector(handle, data.subArray(index(0, column), height), 1);
    }

    /**
     * A copy of this matrix as a 2d cpu array. TDOD: by iterating along columns
     * and then transposing this method can be made faster.
     *
     * @return A copy of this matrix as a 2d cpu array.
     */
    public double[][] get() {
        double[][] getData = new double[height][];
        Arrays.setAll(getData, i -> getRow(i).vecGet());
        return getData;
    }

    /**
     * The trace of this matrix.
     */
    public double getTrace() throws NonSquareMatrixException {
        if (height != width) throw new NonSquareMatrixException(width, height);
        return data.dot(handle, new DSingleton(handle, 1), 0, width + 1);
    }

    /**
     * A hash code for this matrix. This is computed by importing the entire
     * matrix into cpu memory. TODO: do better.
     *
     * @return a hash code for this matrix.
     */
    @Override
    public int hashCode() {
        return new Array2DRowRealMatrix(get()).hashCode();
    }

    /**
     * Created a matrix from a double[] representing a column vector.
     *
     * @param vec The column vector.
     * @param handle
     * @return A matrix representing a column vector.
     */
    public static Matrix fromColVec(double[] vec, Handle handle) {
        Matrix mat = new Matrix(handle, vec.length, 1);
        mat.data.set(handle, vec);
        return mat;
    }

    /**
     * The identity Matrix.
     *
     * @param n the height and width of the matrix.
     * @param hand
     * @param holdIdentity The underlying array that will hold the identity
     * matrix.
     * @return The identity matrix.
     */
    public static Matrix identity(Handle hand, int n, DArray holdIdentity) {

        Matrix ident = new Matrix(hand, holdIdentity, n, n);
        ident.data.fill0(hand);

        ident.data.add(hand, 1, DSingleton.oneOne, 0, n + 1);

        return ident;
    }

    /**
     * The identity Matrix.
     *
     * @param n the height and width of the matrix.
     * @param hand
     * @return The identity matrix.
     */
    public static Matrix identity(Handle hand, int n) {
        return identity(hand, n, DArray.empty(n * n));
    }

    /**
     * Raise this square matrix to a power. This method may use a lot of
     * auxiliary space and allocate new memory multiple times. TODO: fix this.
     *
     * @param p The power.
     * @return This matrix raised to the given power.
     * @throws NotPositiveException
     * @throws NonSquareMatrixException
     */
    public Matrix power(int p) throws NotPositiveException, NonSquareMatrixException {

        if (p < 0) throw new NotPositiveException(p);
        if (height != width) throw new NonSquareMatrixException(width, height);
        if (p == 0) return identity(handle, width);

        if (p % 2 == 0) {
            power(p / 2);
            return Matrix.this.setToProduct(this, this);
        } else {
            try (Matrix copy = copy()) {
                return Matrix.this.setToProduct(copy, power(p - 1));
            }
        }
    }

    /**
     * @see Matrix#setColumnVector(int,
     * org.apache.commons.math3.linear.RealVector)
     *
     * @param column The index of the column to be set.
     * @param vector The vector to be put in the desired location.
     * @throws OutOfRangeException
     * @throws MatrixDimensionMismatchException
     */
    public void setColumnVector(int column, Vector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        data.set(handle, vector.dArray(), index(0, column), 0, 0, vector.colDist, Math.min(height, vector.dim()));
    }

    /**
     * @see Matrix#setRowVector(int, org.apache.commons.math3.linear.RealVector)
     */
    public void setRowVector(int row, Vector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        data.set(handle, vector.dArray(), index(row, 0), 0, colDist, vector.colDist, Math.min(width, vector.dim()));
    }

    /**
     * transposes this matrix.
     *
     * @return
     */
    public Matrix transposeMe() {
        return setSum(true, false, 1, this, 0, this);
    }

    /**
     * A vector containing the dot product of each column and itself.
     *
     * @param workspace Should be as long as the width.
     * @return A vector containing the dot product of each column and itself.
     */
    public Vector columnsSquared(DArray workspace) {

        Vector cs = new Vector(handle, data, 1);

        VectorsStride columns = columns();

        cs.addBatchVecVecMult(1, columns, columns, 0);

        return cs;
    }

    /**
     * The norm of this vector.
     *
     * @param workspace Needs to be width long
     * @return
     */
    public double frobeniusNorm(DArray workspace) {
        return Math.sqrt(columnsSquared(workspace).getL1Norm());

    }

    /**
     * There should be one handle per thread.
     *
     * @param handle The handle used by this matrix.
     */
    public void setHandle(Handle handle) {
        this.handle = handle;
    }

    /**
     * There should be one handle per thread.
     *
     * @return The handle used by this matrix.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * Closes the underlying data of this method.
     */
    @Override
    public void close() {
//        if (colDist != height) {
//            throw new IllegalAccessError("You are cleaning data from a sub Matrix");
//        }
        data.close();
    }

    /**
     * Adds the outer product of a and b to this matrix .
     *
     * @param a A column.
     * @param b A row
     * @return The outer product of a and b.
     */
    public Matrix addOuterProduct(Vector a, Vector b) {

        data.outerProd(handle, height, width, 1, a.dArray(), a.colDist, b.dArray(), b.colDist, colDist);

        return this;
    }

    /**
     * The underlying column major data.
     *
     * @return The underlying column major data.
     */
    public DArray dArray() {
        return data;
    }

    /**
     * Creates a matrix from the underlying data in this matrix with a new
     * height, width, and distance between columns. Note that if the distance
     * between columns in the new matrix is less that in this matrix, the matrix
     * will contain data that this one does not.
     *
     * @param newHieght The height of the new matrix.
     * @param newWidth The width of the new matrix.
     * @param newColDist The distance between columns of the new matrix. By
     * setting the new column distance to be less than or greater than the old
     * one, the new matrix may have more or fewer elements.
     * @return A shallow copy of this matrix that has a different shape.
     *
     */
    public Matrix newDimensions(int newHieght, int newWidth, int newColDist) {
        return new Matrix(handle, data, newHieght, newWidth, newColDist);
    }

    /**
     * Creates a matrix from the underlying data in this matrix with a new
     * height, width, and distance between columns. Note that if the distance
     * between columns in the new matrix is less that in this matrix, the matrix
     * will contain data that this one does not.
     *
     * @param newHieght The height of the new matrix. The width*colDist should
     * be divisible by this number.
     * @return A shallow copy of this matrix that has a different shape.
     *
     */
    public Matrix newDimensions(int newHieght) {
        return newDimensions(newHieght, size() / newHieght, newHieght);
    }

    /**
     * The distance between the 1st element of each column in column major
     * order.
     *
     * @return The distance between the first element of each column in column
     * major order.
     */
    @Override
    public int getColDist() {
        return colDist;
    }

    /**
     * The columns of this matrix.
     *
     * @return The columns of this matrix.
     */
    public VectorsStride columns() {
        return new VectorsStride(handle, data, 1, height, colDist, width);
    }

    /**
     * The rows of this matrix.
     *
     * @return The columns of this matrix.
     */
    public VectorsStride rows() {
        return new VectorsStride(handle, data, colDist, width, 1, height);
    }

    /**
     * A single array containing a copy of the data in this matrix in column
     * major order.
     *
     * @return A single array containing a copy of the data in this matrix in
     * column major order.
     */
    public double[] colMajor() {
        return dArray().get(handle);
    }

    /**
     * This matrix repeating itself in a batch.
     *
     * @param batchSize The size of the batch.
     * @return This matrix repeating itself in a batch.
     */
    public MatricesStride repeating(int batchSize) {
        return new MatricesStride(handle, data, height, width, colDist, 0, batchSize);
    }

    /**
     * This method extracts the lower left corner from this matrix. 1's are
     * place on the diagonal. This is only meant for square matrices that have
     * undergone LU factorization.
     *
     * @param putLHere Where the new matrix is to be placed.
     * @return
     */
    public Matrix lowerLeftUnitDiagonal(DArray putLHere) {
        Matrix l = new Matrix(handle, putLHere, height, width).fill(0);
        for (int i = 0; i < height - 1; i++) {
            l.getColumn(i).getSubVector(i + 1, height - i - 1)
                    .set(getColumn(i).getSubVector(i + 1, height - i - 1));
            l.set(i, i, 1);
        }
        l.set(height - 1, height - 1, 1);
        return l;
    }

    /**
     * This method extracts the upper right corner from this matrix. This is
     * only meant for square matrices that have undergone LU factorization.
     *
     * @param putUHere Where the new matrix is to be placed.
     * @return
     */
    public Matrix upperRight(DArray putUHere) {
        Matrix u = new Matrix(handle, putUHere, height, width).fill(0);
        for (int i = 0; i < height; i++)
            u.getColumn(i).getSubVector(0, i + 1)
                    .set(getColumn(i).getSubVector(0, i + 1));

        return u;
    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                DArray a = new DArray(hand, -1, 2, 3, 2, 4, 5, 3, 5, 6);
                DArray l = DArray.empty(9); DArray u = DArray.empty(9);
                IArray info = IArray.empty(1); IArray pivot = IArray.empty(3);) {

            Matrix m = new Matrix(hand, a, 3, 3);

            m.power(2);
            MatricesStride ms = m.repeating(1);

            System.out.println("m = \n" + m.toString() + "\n");

            Eigen eigen = new Eigen(ms, 1e-13);

            for (int i = 0; i < m.height; i++) {
                double eVal = eigen.values.elmntsAtVecInd(i).get(0);
                Vector eVec = eigen.vectors.getMatrix(0).getColumn(i);

                System.out.println("\nEigen value " + i + ":\n " + eVal);
                System.out.println("Eigen vector " + i + ":\n " + eVec);

                System.out.println("m = \n" + m);

                System.out.println("Checking: is the vector = \n"
                        + eVec.addProduct(
                                false,
                                1 / eVal,
                                m,
                                eVec,
                                0
                        )
                );
            }
        }
    }

    /**
     * The number op non zeroe elements in this matrix.
     *
     * @return The number op non zeroe elements in this matrix.
     */
    public int numNonZeroes() {
        double[] columnMajor = colMajor();
        return (int) Arrays.stream(columnMajor).filter(d -> Math.abs(d) > 1e-10).count();
    }

}
