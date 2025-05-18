package JCudaWrapper.array.Float;

import JCudaWrapper.array.Array;
import static JCudaWrapper.array.Array.UPPER;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasFillMode;
import jcuda.runtime.JCuda;

import JCudaWrapper.array.Array1d;

import jcuda.Sizeof;
import jcuda.jcublas.cublasDiagType;

/**
 *
 * @author E. Dov Neimand
 */
public class FArray1d extends Array1d implements FArray {

    /**
     * An empty array.
     *
     * @param size The number of integers that can be placed in the array.
     */
    public FArray1d(int size) {
        super(size, Sizeof.FLOAT);
    }

    /**
     * Copies from here to there with increments.
     *
     * @param handle The context.
     * @param dst Copy to here.
     */
    public void get(Handle handle, FArray1d dst) {
        opCheck(JCublas2.cublasScopy(
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
     *
     * @param handle The context.
     * @param src The data is copied from here.
     * @return this.
     */
    public FArray1d set(Handle handle, FArray1d src) {
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
    public FArray1d(FArray src, int start, int size) {
        super(src, start, size, 1);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public Singleton get(int index) {
        return new FSingleton(this, index);
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
    public FArray1d(FArray src, int start, int size, int ld) {
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
    public float dot(Handle handle, FArray1d x) {
        float[] result = new float[1];
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
    public void dot(Handle handle, FArray1d x, float[] result, int resultInd) {
        opCheck(JCublas2.cublasSdot(handle.get(), size(), x.pointer(), x.ld(), pointer(), ld(), Pointer.to(result).withByteOffset(resultInd * Sizeof.FLOAT)));

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
    public FArray1d add(Handle handle, float timesX, FArray1d x) {

        confirm(size() == x.size());

        opCheck(JCublas2.cublasSaxpy(handle.get(),
                size(),
                P.to(timesX), x.pointer(), x.ld(),
                pointer(), ld()
        ));

        return this;
    }

    /**
     * Multiplies this array by a scalar.
     *
     * @param handle The context.
     * @param scalar The scalar to multiply this array by.
     * @return this array.
     */
    public FArray1d multiply(Handle handle, float scalar) {

        opCheck(JCublas2.cublasSscal(handle.get(), size(), P.to(scalar), pointer(), ld()));
        return this;
    }

//    /**
//     * {@inheritDoc }
//     */
//    @Override
//    public FArray1d setProduct(Handle handle, float scalar, FArray src) {
//        if (src != this) set(handle, this);
//        return multiply(handle, scalar);
//    }

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
    public float norm(Handle handle, int length, int inc) {
        FSingleton result = new FSingleton();
        norm(handle, length, inc, result);
        return result.getf(handle);
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
    public void norm(Handle handle, int length, int inc, FSingleton result) {

        opCheck(JCublas2.cublasSnrm2(handle.get(), length, pointer(), inc, result.pointer()));
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
    public float sumAbs(Handle handle, int length, int inc) {
        float[] result = new float[1];
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
    public void sumAbs(Handle handle, int length, int inc, float[] result, int toIndex) {

        opCheck(JCublas2.cublasSasum(handle.get(), length, pointer(), inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.FLOAT)));
    }

    /**
     * Performs matrix-vector multiplication with a symmetric banded matrix and
     * updates this vector as:
     *
     * <pre>
     * this = timesA * A * x + timesThis * this
     * </pre>
     *
     * The symmetric banded matrix A is stored compactly in column-major order
     * within {@code Ma}, containing only the nonzero diagonals. The first row
     * of {@code Ma} represents the main diagonal, with subsequent rows storing
     * upper or lower diagonals based on {@code upper}. The leading dimension
     * {@code ldm} defines the row count in this storage.
     *
     * This method uses JCublas `cublasSsbmv` for efficient GPU computation.
     *
     * @param handle The JCublas handle for GPU operations.
     * @param upper Whether the upper triangular part of A is stored.
     * @param colA The order (size) of the symmetric matrix A.
     * @param diagonals The number of sub/superdiagonals in A.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact {@link FArray} representation of A.
     * @param ldm The leading dimension (row count) of {@code Ma}.
     * @param x The input {@link FArray} vector.
     * @param incX Stride for stepping through elements of {@code x}.
     * @param timesThis Scalar multiplier for this vector.
     * @param inc Stride for stepping through elements of this vector.
     * @return This updated vector after the operation.
     */
    public FArray addProductSymBandMatVec(
            Handle handle,
            boolean upper,
            int colA,
            int diagonals,
            float timesA,
            FArray Ma,
            int ldm,
            FArray1d x,
            int incX,
            float timesThis,
            int inc
    ) {
        opCheck(JCublas2.cublasSsbmv(handle.get(),
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
    public FArray1d addProduct(Handle handle, boolean transA, int aRows, int aCols, float timesAx, FArray matA, int lda, FArray1d vecX, int incX, float beta, int inc) {

        opCheck(JCublas2.cublasSgemv(handle.get(),
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
     * Performs matrix-vector multiplication with a banded matrix and updates
     * this vector as:
     *
     * <pre>
     * this = timesAx * op(A) * X + timesThis * this
     * </pre>
     *
     * The banded matrix A, stored compactly in column-major order within
     * {@code Ma}, contains only its nonzero diagonals. The first row of
     * {@code Ma} holds the highest superdiagonal, with subsequent rows storing
     * lower diagonals, down to the lowest subdiagonal. Diagonals that do not
     * extend fully across A are zero-padded. The leading dimension {@code ldm}
     * defines the row count in this storage.
     *
     * This method uses JCublas `cublasSgbmv` for efficient GPU computation.
     * When using a 3D array with a pitch, ensure {@code ld} accounts for it:
     * <pre>
     * int ld = (int)(pitchedPtr.pitch / bytesPerElement);
     * </pre>
     *
     * @param handle The JCublas handle for GPU operations.
     * @param transposeA Whether to transpose A before multiplying.
     * @param rowsA Number of rows in A.
     * @param colsA Number of columns in A.
     * @param subDiagonalsA Number of subdiagonals in A.
     * @param superDiagonalA Number of superdiagonals in A.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma Compact {@link FArray} representation of A.
     * @param ldm Leading dimension (row count) of {@code Ma}.
     * @param x Input {@link FArray} vector.
     * @param incX Stride for stepping through elements of {@code x}.
     * @param timesThis Scalar multiplier for this vector.
     * @param inc Stride for stepping through elements of this vector.
     * @return This updated vector after the operation.
     */
    public FArray1d addProductBandMatVec(Handle handle, boolean transposeA, int rowsA, int colsA, int subDiagonalsA, int superDiagonalA, float timesA, FArray Ma, int ldm, FArray x, int incX, float timesThis, int inc) {
        opCheck(JCublas2.cublasSgbmv(handle.get(),
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
     * Performs matrix-vector multiplication with a symmetric banded matrix and 
     * updates this vector as:
     *
     * <pre>
     * this = timesA * A * x + timesThis * this
     * </pre>
     *
     * The symmetric banded matrix A, stored compactly in {@code Ma}, retains 
     * only its upper (or lower) part, reducing memory usage. The first row of 
     * {@code Ma} contains the main diagonal, with subsequent rows storing 
     * superdiagonals (if {@code upper} is true) or subdiagonals otherwise. 
     * {@code ldm} defines the leading dimension (row count) in this storage.
     *
     * This method uses JCublas `cublasSsbmv` for efficient GPU computation.
     *
     * @param handle The JCublas handle for GPU operations.
     * @param upper Whether the upper triangular part of A is stored.
     * @param colA Order of the symmetric matrix A (number of rows/columns).
     * @param diagonals Number of subdiagonals or superdiagonals.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma Compact {@link FArray3d} representation of A.
     * @param ldm Leading dimension (row count) of {@code Ma}.
     * @param x Input {@link FArray3d} vector.
     * @param incX Stride for stepping through elements of {@code x}.
     * @param timesThis Scalar multiplier for this vector.
     * @param inc Stride for stepping through elements of this vector.
     * @return This updated vector after the operation.
     */

    public FArray1d addProductSymBandMatVec(Handle handle, boolean upper, int colA, int diagonals, float timesA, FArray Ma, int ldm, FArray x, int incX, float timesThis, int inc) {
        opCheck(JCublas2.cublasSsbmv(handle.get(),
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
     * This method uses JCublas `cublasStbsv` function to solve the system of
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
     * ({@code cublasSiagType.CUBLAS_DIAG_UNIT}) or non-unit triangular
     * ({@code cublasSiagType.CUBLAS_DIAG_NON_UNIT}).
     * @param rowsA The number of rows/columns of the matrix A (the order of the
     * matrix).
     * @param nonPrimaryDiagonals The number of subdiagonals or superdiagonals
     * in the triangular banded matrix.
     * @param Ma A compact form {@link FArray} representing the triangular
     * banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted matrix representation.
     *
     * @param inc The stride for stepping through the elements of b.
     * @return The updated {@link FArray} (b), now containing the solution
     * vector x.
     */
    public FArray solveTriangularBandedSystem(Handle handle, boolean isUpper, boolean transposeA, boolean onesOnDiagonal, int rowsA, int nonPrimaryDiagonals, FArray Ma, int ldm, int inc) {
        // Call the cublasStbsv function to solve the system
        opCheck(JCublas2.cublasStbsv(
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
        return new FArray1d(size()).set(handle, this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray2d as2d(int entriesPerLine) {
        return new FArray2d(this, entriesPerLine);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array2d as2d(int entriesPerLine, int ld) {
        return new FArray2d(this, entriesPerLine, ld);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray3d as3d(int entriesPerLine, int linesPerLayer) {
        return new FArray3d(this, entriesPerLine, linesPerLayer);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray3d as3d(int entriesPerLine, int ld, int linesPerLayer) {
        return new FArray3d(this, entriesPerLine, ld, linesPerLayer);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray1d set(Handle handle, Array from) {
        super.set(handle, from);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FArray1d set(Handle handle, float... srcCPUArray) {
        FArray.super.set(handle, srcCPUArray);
        return this;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray1d sub(int start, int length) {
        return new FArray1d(this, start, length);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public FArray1d sub(int start, int length, int ld) {
        return new FArray1d(this, start, length, ld);
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
    public float[] get(Handle handle) {

        float[] cpuArray = new float[size()];

        if (!hasPadding()) get(handle, Pointer.to(cpuArray));
        else try (FArray1d temp = new FArray1d(size())) {
            get(handle, temp);
            temp.get(handle, Pointer.to(cpuArray));
        }

        return cpuArray;

    }
}
