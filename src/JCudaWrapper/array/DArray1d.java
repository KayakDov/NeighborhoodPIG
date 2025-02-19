package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasDiagType;
import jcuda.jcublas.cublasFillMode;
import jcuda.runtime.JCuda;

/**
 *
 * @author E. Dov Neimand
 */
public class DArray1d extends Array1d implements DArray {

    /**
     * An empty array.
     *
     * @param size The number of integers that can be placed in the array.
     */
    public DArray1d(int size) {
        super(size, Sizeof.DOUBLE);
    }

    /**
     * Copies from here to there with increments.
     *
     * @param handle The context.
     * @param dst Copy to here.
     */
    public void get(Handle handle, DArray1d dst) {
        opCheck(JCublas2.cublasDcopy(
                handle.get(),
                Math.min(size(), dst.size()),
                pointer(),
                ld(),
                dst.pointer(),
                dst.ld()
        ));
    }

    /**
     * Copies the data from the src to here taking into acount increments.
     * @param handle The context.
     * @param src The data is copied from here.
     * @return this.
     */
    public DArray1d set(Handle handle, DArray1d src) {
        src.get(handle, this);
        return this;
    }

    
    
    /**
     * Constructs a 1d sub array of the proffered array. If the array copied
     * from is not 1d, then depending on the length, this array may include
     * pitch.
     *
     * @param src The array to be copied from.
     * @param start The start index of the array.
     * @param size The length of the array.
     */
    public DArray1d(DArray src, int start, int size) {
        super(src, start, size, 1);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Singleton get(int index) {
        return new DSingleton(this, index);
    }

    /**
     * Constructs a 1d sub array of the proffered array. If the array copied
     * from is not 1d, then depending on the length, this array may include
     * pitch.
     *
     * @param src The array to be copied from.
     * @param start The start index of the array.
     * @param size The length of the array.
     * @param ld The increment.
     */
    public DArray1d(DArray src, int start, int size, int ld) {
        super(src, start, size, ld);
    }

    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param x Pointer to vector X in GPU memory. If this is not a one
     * dimensional array, then the increment should account for that.
     * @return The dot product of X and Y.
     */
    public double dot(Handle handle, DArray1d x) {
        double[] result = new double[1];
        dot(handle, x, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param x Pointer to vector X in GPU memory. If this is not a one
     * dimensional array then the increment should account for that.
     * @param result The array the answer should be put in.
     * @param resultInd The index of the array the answer should be put in.
     */
    public void dot(Handle handle, DArray1d x, double[] result, int resultInd) {
        opCheck(JCublas2.cublasDdot(handle.get(), size(), x.pointer(), x.ld(), pointer(), ld(), Pointer.to(result).withByteOffset(resultInd * Sizeof.DOUBLE)));

    }

    /**
     * Performs the vector addition:
     *
     * <pre>
     * this = timesX * X + this
     * </pre>
     *
     * This operation scales vector X by alpha and adds it to vector Y.
     *
     * @param handle handle to the cuBLAS library context.
     * @param timesX Scalar used to scale vector X.
     * @param x Pointer to vector X in GPU memory.
     * @return this
     */
    public DArray1d add(Handle handle, double timesX, DArray1d x) {

        confirm(size() == x.size());

        opCheck(JCublas2.cublasDaxpy(handle.get(),
                size(),
                P.to(timesX), x.pointer(), x.ld(),
                pointer(), ld()
        ));

        return this;
    }

    /**
     * Multiplies this array by a scalar.
     * @param handle The context.
     * @param scalar The scalar to multiply this array by.
     * @return this array.
     */
    public DArray1d multiply(Handle handle, double scalar) {
        
        opCheck(JCublas2.cublasDscal(handle.get(), size(), P.to(scalar), pointer(), ld()));
        return this;
    }
    
    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d setProduct(Handle handle, double scalar, DArray src) {
        if(src != this) set(handle, this);
        return multiply(handle, scalar);
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm): This method
     * synchronizes the handle.
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars that will be squared.
     * @param inc The stride step over this array.
     * @return The Euclidean norm of this vector.
     */
    public double norm(Handle handle, int length, int inc) {
        DSingleton result = new DSingleton();
        norm(handle, length, inc, result);
        return result.getVal(handle);
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm):
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars that will be squared.
     * @param inc The stride step over this array.
     * @param result where the result is to be stored.
     */
    public void norm(Handle handle, int length, int inc, DSingleton result) {

        opCheck(JCublas2.cublasDnrm2(handle.get(), length, pointer(), inc, result.pointer()));
    }

    /**
     * Finds the index of the element with the minimum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @param result where the result (index of the minimum absolute value) is
     * to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void argMinAbs(Handle handle, int length, int inc, int[] result, int toIndex) {

        opCheck(JCublas2.cublasIdamin(handle.get(), length, pointer(), inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.INT)));
        result[toIndex] -= 1;//It looks like the cuda methods are index-1 based.
    }

    /**
     * Finds the index of the element with the maximum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param result where the result (index of the maximum absolute value) is
     * to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void argMaxAbs(Handle handle, int[] result, int toIndex) {

        opCheck(JCublas2.cublasIdamax(
                handle.get(),
                size(),
                pointer(),
                ld(),
                Pointer.to(result).withByteOffset(toIndex * Sizeof.INT)
        ));
        result[toIndex] -= 1; //It looks like the cuda methods are index-1 based.
    }

    /**
     * Finds the index of the element with the minimum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @return The index of the element with the minimum absolute value.
     */
    public int argMinAbs(Handle handle) {
        int[] result = new int[1];
        argMinAbs(handle, size(), ld(), result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Finds the index of the element with the maximum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of max(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @return The index of the element with greatest absolute value.
     *
     */
    public int argMaxAbs(Handle handle) {
        int[] result = new int[1];
        argMaxAbs(handle, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the sum of the absolute values of the vector X (1-norm):
     *
     * <pre>
     * result = |X[0]| + |X[1]| + ... + |X[n-1]|
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars to include in the sum.
     * @param inc The stride step over the array.
     * @return The l1 norm of the vector.
     */
    public double sumAbs(Handle handle, int length, int inc) {
        double[] result = new double[1];
        sumAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the sum of the absolute values of the vector X (1-norm):
     *
     * <pre>
     * result = |X[0]| + |X[1]| + ... + |X[n-1]|
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars to include in the sum.
     * @param inc The stride step over the array.
     * @param result where the result is to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void sumAbs(Handle handle, int length, int inc, double[] result, int toIndex) {

        opCheck(JCublas2.cublasDasum(handle.get(), length, pointer(), inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.DOUBLE)));
    }

    /**
     * Multiplies this vector by a symmetric banded matrix and adds the result
     * to the vector.
     *
     * this = timesA * A * x + timesThis * this
     *
     * A symmetric banded matrix is a matrix where the non-zero elements are
     * confined to a diagonal band around the main diagonal, and the matrix is
     * symmetric (i.e., A[i][j] = A[j][i]). In a symmetric banded matrix, only
     * the elements within the band are stored, as the symmetry allows the upper
     * or lower part to be inferred. This storage technique reduces memory
     * usage.
     *
     * In this method, the symmetric banded matrix is represented by the A
     * stored in Ma, where only the upper (or lower) part of the matrix is
     * stored. The matrix is stored in a compact form, with each column being a
     * segment of the band. The parameter `ldm` indicates the leading dimension
     * of the matrix, which corresponds to the number of rows in the compacted
     * matrix representation. Only the non-zero diagonals of the matrix are
     * stored contiguously in memory.
     *
     * Let M represent the column-major matrix that stores the elements of the
     * symmetric banded matrix A in a {@link DArray}. The first row of M
     * corresponds to the main diagonal of A, and the subsequent rows correspond
     * to diagonals above or below the main diagonal. For instance, the second
     * row corresponds to the diagonal directly above the main diagonal, and so
     * on.
     *
     * This method performs the matrix-vector multiplication between the
     * symmetric banded matrix A and the vector x using the JCublas
     * `cublasDsbmv` function, which supports operations on symmetric banded
     * matrices. The result is scaled by `timesA` and added to this vector
     * scaled by `timesThis`.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param upper Whether the upper triangular part of the matrix is stored.
     * @param colA The order of the symmetric matrix A (number of rows and
     * columns).
     * @param diagonals The number of subdiagonals or superdiagonals in the
     * matrix.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray} representing the symmetric banded
     * matrix A.
     * @param ldm The leading dimension of the matrix, defining the row count in
     * the compacted matrix representation.
     * @param x The {@link DArray} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray addProductSymBandMatVec(
            Handle handle,
            boolean upper,
            int colA,
            int diagonals,
            double timesA,
            DArray Ma,
            int ldm,
            DArray1d x,
            int incX,
            double timesThis,
            int inc
    ) {
        opCheck(JCublas2.cublasDsbmv(handle.get(),
                upper ? cublasFillMode.CUBLAS_FILL_MODE_UPPER : cublasFillMode.CUBLAS_FILL_MODE_LOWER,
                colA,
                diagonals,
                P.to(timesA),
                Ma.pointer(), ldm,
                x.pointer(),
                incX,
                P.to(timesThis),
                pointer(),
                inc
        ));
        return this;
    }

    /**
     * Performs the matrix-vector multiplication:
     *
     * <pre>
     * this = timesAx * op(A) * X + beta * this
     * </pre>
     *
     * Where op(A) can be A or its transpose.
     *
     * @param handle handle to the cuBLAS library context.
     * @param transA Specifies whether matrix A is transposed (true for
     * transpose and false for not.)
     * @param aRows The number of rows in matrix A.
     * @param aCols The number of columns in matrix A.
     * @param timesAx Scalar multiplier applied to the matrix-vector product.
     * @param matA Pointer to matrix A in GPU memory.
     * @param lda The distance between the first element of each column of A.
     * @param vecX Pointer to vector X in GPU memory.
     * @param incX The increments taken when iterating over elements of X. This
     * is usually1 1. If you set it to 2 then you'll be looking at half the
     * elements of x.
     * @param beta Scalar multiplier applied to vector Y before adding the
     * matrix-vector product.
     * @param inc the increment taken when iterating over elements of this
     * array.
     * @return this array after this = timesAx * op(A) * X + beta*this
     */
    public DArray1d addProduct(Handle handle, boolean transA, int aRows, int aCols, double timesAx, DArray matA, int lda, DArray1d vecX, int incX, double beta, int inc) {

        opCheck(JCublas2.cublasDgemv(handle.get(),
                Array.transpose(transA),
                aRows, aCols,
                P.to(timesAx),
                matA.pointer(), lda,
                vecX.pointer(), incX,
                P.to(beta),
                pointer(),
                inc
        ));
        return this;
    }

    /**
     * Multiplies this vector by a banded matrix and adds the result to the
     * vector.
     *
     * this = timesAx * op(A) * X + timesThis * this
     *
     * A banded matrix is a sparse matrix where the non-zero elements are
     * confined to a diagonal band, comprising the main diagonal, a fixed number
     * of subdiagonals below the main diagonal, and a fixed number of
     * superdiagonals above the main diagonal. Banded matrices are often used to
     * save space, as only the elements within the band are stored, and the rest
     * are implicitly zero.
     *
     * In this method, the banded matrix is represented by the {@link DArray} M,
     * and the structure of M is defined by the number of subdiagonals and
     * superdiagonals. The matrix is stored in a column-major order, with each
     * column being a segment of the band. The parameter `lda` indicates the
     * leading dimension of the banded matrix, which corresponds to the number
     * of rows in the compacted matrix representation. The elements of the band
     * are stored contiguously in memory, with zero-padding where necessary to
     * fill out the bandwidth of the matrix.
     *
     * Let M represent the column-major matrix that stores the elements of A in
     * a {@link DArray}. The first row of M corresponds to the top-rightmost
     * non-zero diagonal of A (the highest superdiagonal). The second row
     * corresponds to the diagonal that is one position below/left of the first
     * row, and so on, proceeding down the diagonals. The final row of M
     * contains the bottom-leftmost diagonal of A (the lowest subdiagonal).
     * Diagonals that do not fully extend across A are padded with zeros in M.
     * An element in A has the same column in M as it does in A.
     *
     * This method performs the matrix-vector multiplication between the banded
     * matrix A and the vector x using the JCublas `cublasDgbmv` function, which
     * supports operations on banded matrices. The result is scaled by `timesA`
     * and added to this vector scaled by `timesThis`.
     *
     * if using a 3d array with a pitch, be sure to account for that with ld.
     * int ld = (int)(pitchedPtr.pitch / bytesPerElement);
     *
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param transposeA Whether to transpose matrix A before multiplying.
     * @param rowsA The number of rows in matrix A.
     * @param colsA The number of columns in matrix A.
     * @param subDiagonalsA The number of subdiagonals in matrix A.
     * @param superDiagonalA The number of superdiagonals in matrix A.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray} representing the banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted banded matrix representation.
     * @param x The {@link DArray} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray1d addProductBandMatVec(Handle handle, boolean transposeA, int rowsA, int colsA, int subDiagonalsA, int superDiagonalA, double timesA, DArray Ma, int ldm, DArray x, int incX, double timesThis, int inc) {
        opCheck(JCublas2.cublasDgbmv(handle.get(),
                Array.transpose(transposeA),
                rowsA, colsA,
                subDiagonalsA, superDiagonalA,
                P.to(timesA),
                Ma.pointer(), ldm,
                x.pointer(), incX,
                P.to(timesThis), pointer(), inc));

        return this;
    }

    /**
     * Multiplies this vector by a symmetric banded matrix and adds the result
     * to the vector.
     *
     * this = timesA * A * x + timesThis * this
     *
     * A symmetric banded matrix is a matrix where the non-zero elements are
     * confined to a diagonal band around the main diagonal, and the matrix is
     * symmetric (i.e., A[i][j] = A[j][i]). In a symmetric banded matrix, only
     * the elements within the band are stored, as the symmetry allows the upper
     * or lower part to be inferred. This storage technique reduces memory
     * usage.
     *
     * In this method, the symmetric banded matrix is represented by the A
     * stored in Ma, where only the upper (or lower) part of the matrix is
     * stored. The matrix is stored in a compact form, with each column being a
     * segment of the band. The parameter `ldm` indicates the leading dimension
     * of the matrix, which corresponds to the number of rows in the compacted
     * matrix representation. Only the non-zero diagonals of the matrix are
     * stored contiguously in memory.
     *
     * Let M represent the column-major matrix that stores the elements of the
     * symmetric banded matrix A in a {@link DArray3d}. The first row of M
     * corresponds to the main diagonal of A, and the subsequent rows correspond
     * to diagonals above or below the main diagonal. For instance, the second
     * row corresponds to the diagonal directly above the main diagonal, and so
     * on.
     *
     * This method performs the matrix-vector multiplication between the
     * symmetric banded matrix A and the vector x using the JCublas
     * `cublasDsbmv` function, which supports operations on symmetric banded
     * matrices. The result is scaled by `timesA` and added to this vector
     * scaled by `timesThis`.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param upper Whether the upper triangular part of the matrix is stored.
     * @param colA The order of the symmetric matrix A (number of rows and
     * columns).
     * @param diagonals The number of subdiagonals or superdiagonals in the
     * matrix.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray3d} representing the symmetric
     * banded matrix A.
     * @param ldm The leading dimension of the matrix, defining the row count in
     * the compacted matrix representation.
     * @param x The {@link DArray3d} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray1d addProductSymBandMatVec(Handle handle, boolean upper, int colA, int diagonals, double timesA, DArray Ma, int ldm, DArray x, int incX, double timesThis, int inc) {
        opCheck(JCublas2.cublasDsbmv(handle.get(),
                upper ? Array.UPPER : Array.LOWER,
                colA,
                diagonals,
                P.to(timesA),
                Ma.pointer(), ldm,
                x.pointer(),
                incX,
                P.to(timesThis),
                pointer(),
                inc));

        return this;
    }

    /**
     * Solves the system of linear equations Ax = b, where A is a triangular
     * banded matrix, and x is the solution vector.
     *
     * b is this when the algorithm begins, and x is this when the algorithm
     * ends. That is to say the solution to Ax = this is stored in this.
     *
     * A triangular banded matrix is a special type of sparse matrix where the
     * non-zero elements are confined to a diagonal band around the main
     * diagonal and only the elements above (or below) the diagonal are stored,
     * depending on whether the matrix is upper or lower triangular.
     *
     * The matrix is stored in a compact banded format, where only the diagonals
     * of interest are represented to save space. For a lower diagonal matrix,
     * the first row represents the main diagonal, and subsequent rows represent
     * the diagonals progressively further from the main diagonal. An upper
     * diagonal matrix is stored with the last row as the main diagonal and the
     * first row as the furthest diagonal from the main.
     *
     * This method uses JCublas `cublasDtbsv` function to solve the system of
     * equations for the vector x.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param isUpper Indicates whether the matrix A is upper or lower
     * triangular. Use {@code cublasFillMode.CUBLAS_FILL_MODE_UPPER} for upper
     * triangular, or {@code cublasFillMode.CUBLAS_FILL_MODE_LOWER} for lower
     * triangular.
     * @param transposeA Whether to transpose the matrix A before solving the
     * system. Use {@code cublasOperation.CUBLAS_OP_T} for transpose, or
     * {@code cublasOperation.CUBLAS_OP_N} for no transpose.
     * @param onesOnDiagonal Specifies whether the matrix A is unit triangular
     * ({@code cublasDiagType.CUBLAS_DIAG_UNIT}) or non-unit triangular
     * ({@code cublasDiagType.CUBLAS_DIAG_NON_UNIT}).
     * @param rowsA The number of rows/columns of the matrix A (the order of the
     * matrix).
     * @param nonPrimaryDiagonals The number of subdiagonals or superdiagonals
     * in the triangular banded matrix.
     * @param Ma A compact form {@link DArray} representing the triangular
     * banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted matrix representation.
     *
     * @param inc The stride for stepping through the elements of b.
     * @return The updated {@link DArray} (b), now containing the solution
     * vector x.
     */
    public DArray solveTriangularBandedSystem(Handle handle, boolean isUpper, boolean transposeA, boolean onesOnDiagonal, int rowsA, int nonPrimaryDiagonals, DArray Ma, int ldm, int inc) {
        // Call the cublasDtbsv function to solve the system
        opCheck(JCublas2.cublasDtbsv(
                handle.get(),
                isUpper ? UPPER : cublasFillMode.CUBLAS_FILL_MODE_LOWER, // Upper or lower triangular matrix
                Array.transpose(transposeA),
                onesOnDiagonal ? cublasDiagType.CUBLAS_DIAG_UNIT : cublasDiagType.CUBLAS_DIAG_NON_UNIT, // Whether A is unit or non-unit triangular
                rowsA, // Number of rows/columns in A
                nonPrimaryDiagonals, // Number of subdiagonals/superdiagonals
                Ma.pointer(), // Pointer to the compact form of matrix A
                ldm, // Leading dimension of Ma
                pointer(), // Pointer to the right-hand side vector (b)
                inc));          // Stride through the elements of b

        return this;  // The result (solution vector x) is stored in b
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array copy(Handle handle) {
        return new DArray1d(size()).set(handle, this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray2d as2d(int entriesPerLine) {
        return new DArray2d(this, entriesPerLine);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        return new DArray2d(this, entriesPerLine, ld);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray3d as3d(int entriesPerLine, int linesPerLayer) {
        return new DArray3d(this, entriesPerLine, linesPerLayer);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        return new DArray3d(this, entriesPerLine, ld, linesPerLayer);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray1d set(Handle handle, Array from) {
        super.set(handle, from);
        return this;
    }
    

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray1d set(Handle handle, double... srcCPUArray) {
        DArray.super.set(handle, srcCPUArray);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d sub(int start, int length) {
        return new DArray1d(this, start, length);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray1d sub(int start, int length, int ld) {
        return new DArray1d(this, start, length, ld);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        JCuda.cudaDeviceSynchronize();
        try (Handle handle = new Handle()) {
            return Arrays.toString(get(handle));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] get(Handle handle) {

        double[] cpuArray = new double[size()];

        if (!hasPadding()) get(handle, Pointer.to(cpuArray));
        else try (DArray1d temp = new DArray1d(size())) {
            get(handle, temp);
            temp.get(handle, Pointer.to(cpuArray));
        }
        
        return cpuArray;

    }
}
