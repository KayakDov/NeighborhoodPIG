package JCudaWrapper.array;

import jcuda.jcublas.JCublas2;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverStatus;

/**
 * Class for managing a batched 2D array of arrays (DArrays) on the GPU and
 * supporting various operations including batched matrix-matrix multiplication.
 *
 * Provides support for batched matrix-matrix multiplication using the cuBLAS
 * library.
 *
 * @author E. Dov Neimand
 */
public interface DPointerArray extends PointerArray {

    /**
     * Performs batched matrix-matrix multiplication:
     *
     * <pre>
     * Result[i] = timesAB * op(A[i]) * op(B[i]) + timesResult * this[i]
     * </pre>
     *
     * Where op(A) and op(B) can be A and B or their transposes.
     *
     * This method computes multiple matrix-matrix multiplications at once
     * without using strided data access, i.e., it processes independent
     * batches.
     *
     * @param handle Handle to the cuBLAS library context.
     * @param transA True if matrix A should be transposed, false otherwise.
     * @param transB True if matrix B should be transposed, false otherwise.
     * @param aRows The number of rows in matrix A.
     * @param aColsBRows The number of columns in matrix A and the number of
     * rows in matrix B.
     * @param bCols The number of columns in matrix B.
     * @param timesAB Scalar multiplier applied to the matrix-matrix product.
     * @param A Array of pointers to matrices A (in GPU memory).
     * @param lda Leading dimension of each matrix A (number of elements between
     * consecutive columns in memory).
     * @param B Array of pointers to matrices B (in GPU memory).
     * @param ldb Leading dimension of each matrix B (number of elements between
     * consecutive columns in memory).
     * @param timesResult Scalar multiplier applied to each result matrix before
     * adding the matrix-matrix product.
     * @param ldResult Leading dimension of each result matrix (number of
     * elements between consecutive columns in memory).
     * @param batchCount The number of matrix-matrix multiplications to compute.
     */
    public default void addProductBatched(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DPointerArray A,
            int lda, DPointerArray B, int ldb, double timesResult, int ldResult, int batchCount) {

        // Perform the batched matrix-matrix multiplication using cuBLAS
        JCublas2.cublasDgemmBatched(
                handle.get(), // cuBLAS handle
                Array.transpose(transA),
                Array.transpose(transB),
                aRows, bCols, aColsBRows, // Number of columns of A / rows of B
                P.to(timesAB),
                A.pointer(), lda, // Leading dimension of A
                B.pointer(), ldb, // Leading dimension of B
                P.to(timesResult), pointer(), ldResult, // Leading dimension of result matrices
                batchCount // Number of matrices to multiply
        );
    }
    
    /**
     * Performs batched eigenvector computation for symmetric matrices.
     *
     * This function computes the Cholesky factorization of a sequence of
     * Hermitian positive-definite matrices.
     *
     *
     * If input parameter fill is LOWER, only lower triangular part of A is
     * processed, and replaced by lower triangular Cholesky factor L.
     *
     *
     * If input parameter uplo is UPPER, only upper triangular part of A is
     * processed, and replaced by upper triangular Cholesky factor U. * Remark:
     * the other part of A is used as a workspace. For example, if uplo is
     * CUBLAS_FILL_MODE_UPPER, upper triangle of A contains Cholesky factor U
     * and lower triangle of A is destroyed after potrfBatched.
     *
     * @param handle Handle to cuSolver context.
     * @param n The dimension of the symmetric matrices.
     * @param lda Leading dimension of matrix A (n).
     * @param fill The part of the dense matrix that is looked at and replaced.
     * @param info infoArray is an integer array of size batchsize. If
     * potrfBatched returns CUSOLVER_STATUS_INVALID_VALUE, infoArray[0] = -i
     * (less than zero), meaning that the i-th parameter is wrong (not counting
     * handle). If potrfBatched returns CUSOLVER_STATUS_SUCCESS but infoArray[i]
     * = k is positive, then i-th matrix is not positive definite and the
     * Cholesky factorization failed at row k.
     */
    public default void choleskyFactorization(Handle handle, int n, int lda, IArray info, int fill) {

        int success = JCusolverDn.cusolverDnDpotrfBatched(
                handle.solverHandle(), // cuSolver handle
                fill,
                n, // Matrix dimension
                pointer(), // Input matrices (symmetric)
                lda, // Leading dimension                
                info.pointer(), // Info array for errors
                size() // Number of matrices
        );
        if (success != cusolverStatus.CUSOLVER_STATUS_SUCCESS)
            throw new RuntimeException("choleskyFactorization didn't work.  "
                    + "Here's the info array: " + info.toString());
    }

    /**
     * Performs batched LU factorization of general matrices.
     * https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrfbatched This
     * function computes the LU factorization of a sequence of general matrices
     * A. It factors each matrix A as A = P * L * U, where P is a permutation
     * matrix, L is lower triangular (or unit lower triangular in the case of
     * square matrices), and U is upper triangular.
     *
     * The factorization is stored in place. The pivot array contains the pivot
     * indices for each matrix.
     *
     * @param handle Handle to cuSolver context.
     * @param rowsAndColumnsThis The dimension of the matrices (nxn).
     * @param ldThis Leading dimension of matrix A.
     * @param pivotArray Array of pivots, where pivotArray[i] contains the pivot
     * indices for matrix i. This is an output array and whatever is in it will
     * be overwritten.
     * @param infoArray Info array for status/error reporting, where
     * infoArray[i] contains info for the i-th matrix. If infoArray[i] > 0, the
     * matrix is singular.
     */
    public default void luFactor(Handle handle, int rowsAndColumnsThis, int ldThis, IArray pivotArray, IArray infoArray) {

        opCheck(JCublas2.cublasDgetrfBatched(
                handle.get(), // cuSolver handle
                rowsAndColumnsThis, // Matrix dimension
                pointer(), // Input matrices (general)
                ldThis, // Leading dimension
                pivotArray.pointer(), // Pivot indices for each matrix
                infoArray.pointer(), // Info array for errors
                size() // Number of matrices in the batch
        ));
    }

    /**
     * Solves a system of linear equations using the LU factorization obtained
     * by `getrfBatched`.
     * https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrsbatched This
     * function solves a system of linear equations A * X = B for each matrix A
     * in the batch. It uses the LU decomposition from `getrfBatched` to find
     * the solution.
     *
     * @param handle Handle to cuBLAS context.
     * @param transposeA true if this matrix should be transposed. False
     * otherwise.
     * @param colsAndRowsA The dimension of the matrices (nxn).
     * @param colsB The number of right-hand sides, i.e., the number of columns
     * of the matrix B.
     * @param ldThis Leading dimension of matrix A.
     * @param ldb Leading dimension of matrix B.
     * @param pivotArray Pivot indices as generated by `getrfBatched`.
     * @param B The right-hand side matrices B. On exit, contains the solutions
     * X.
     * @param info Info array for status/error reporting. If infoArray[i] > 0,
     * matrix i is singular.
     */
    public default void solveWithLUFactored(Handle handle, boolean transposeA, int colsAndRowsA,
            int colsB, int ldThis, int ldb, IArray pivotArray, DPointerArray B, IArray info) {

        opCheck(JCublas2.cublasDgetrsBatched(
                handle.get(),
                Array.transpose(transposeA),
                colsAndRowsA, colsB,
                pointer(), ldThis,
                pivotArray.pointer(), // Pivot indices for each matrix
                B.pointer(), ldb,
                info.pointer(),
                size() // Number of matrices in the batch
        ));
    }

    /**
     * Solves a symmetric positive definite system of linear equations A * x =
     * b, where A is a symmetric matrix that has undergone Cholesky
     * factorization and B and X are matrices of right-hand side vectors and
     * solutions, respectively.
     *
     * This method utilizes the cuSolver library and the
     * `cusolverDnDpotrsBatched` function to solve a batch of systems using the
     * Cholesky factorization. The matrix A must be symmetric positive definite.
     *
     * The input matrix A is provided in packed format, with either the upper or
     * lower triangular part of the matrix being supplied based on the `fillA`
     * parameter.
     *
     * This method checks for valid inputs and initializes the info array if not
     * provided. The `info` array stores error messages for each matrix in the
     * batch.
     *
     * @param handle The cuSolver handle, which is required for cuSolver library
     * operations. Must not be null.
     * @param fillA Indicates whether the upper or lower triangular part of A is
     * stored. It should be either {@link Fill#UPPER} or {@link Fill#LOWER}.
     * @param heightA The number of rows of the matrix A.
     * @param lda The leading dimension of the matrix A.
     * @param b The right-hand side matrix B and will store the solution matrix
     * X after computation. The input matrix must be stored in column-major
     * order on the GPU.
     * @param ldb The leading dimension of the matrix B.
     * @param info An optional output array to store the status of each system
     * in the batch. If `info == null`, an array will be created internally. If
     * info is not null, it must have a length equal to the number of matrices
     * in the batch.
     *
     * @throws IllegalArgumentException if the handle, fillA, or b is null.
     * @throws IllegalArgumentException if any of the dimensions (heightA,
     * widthBAndX, lda, ldb) are not positive.
     */
    public default void solveCholesky(Handle handle, int fillA, int heightA, int lda, DPointerArray b, int ldb, IArray info) {

        boolean cleanInfo = false;
        if (info == null) {
            info = new IArray1d(0);

            cleanInfo = true;
        }
        JCusolverDn.cusolverDnDpotrsBatched(
                handle.solverHandle(), fillA,
                heightA, 1,
                pointer(), lda,
                b.pointer(), ldb,
                info.pointer(),
                size()
        );
        if (cleanInfo) info.close();
    }

    /**
     * Fills this arrays with pointers to the given array.
     *
     * @param handle
     * @param source The array pointed to,
     * @param inc The distance between the pointer targets.
     * @return this
     */
    public default DPointerArray fill(Handle handle, DArray source, int inc) {
        Kernel.run("genPtrs", handle, size(), source, P.to(inc), P.to(this), P.to(1));
        return this;
    }

    /**
     * Fills this arrays with pointers to the given array.
     *
     * @param handle
     * @param source The array pointed to,
     * @return this
     */
    public default DPointerArray fill(Handle handle, DStrideArray source) {
        return fill(handle, source, source.stride());
    }

}
