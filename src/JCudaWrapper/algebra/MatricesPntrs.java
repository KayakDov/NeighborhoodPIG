package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import java.awt.Point;
import java.util.function.Consumer;

import org.apache.commons.math3.exception.DimensionMismatchException;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;

/**
 * A class representing a batch of matrices stored in GPU memory, allowing
 * efficient operations such as batched matrix-matrix multiplications.
 *
 * <p>
 * This class provides a wrapper around the underlying {@link DPointerArray} data
 * structure and offers batch matrix operations that utilize the GPU for high
 * performance.
 * </p>
 *
 * <p>
 * Each matrix in the batch is expected to have the same dimensions, and the
 * column distance (stride) between matrix elements can be controlled via the
 * {@code colDist} parameter.
 * </p>
 *
 * <p>
 * The class supports optional transposition of matrices during operations via
 * the {@code transposeForOperations} flag.
 * </p>
 *
 * @author E. Dov Neimand
 */
public class MatricesPntrs implements AutoCloseable {

    /**
     * The number of rows (height) of each matrix in the batch.
     */
    private final int height;

    /**
     * The number of columns (width) of each matrix in the batch.
     */
    private final int width;

    /**
     * The number of elements between the first element of each column of each
     * matrix (column stride or leading dimension).
     */
    private final int colDist;

    /**
     * The underlying data structure that stores the batch of matrices in GPU
     * memory.
     */
    private final DPointerArray arrays;

    /**
     * Indicates whether the matrices should be transposed during operations. If
     * true, matrices are transposed before being used in operations.
     */
    public boolean transposeForOperations;

    /**
     * Constructs a batch of matrices.
     *
     * @param height The height (number of rows) of each matrix.
     * @param width The width (number of columns) of each matrix.
     * @param colDist The number of elements between the first element of each
     * column (column stride or leading dimension).
     * @param arrays The underlying data arrays that store the batch of
     * matrices.
     */
    public MatricesPntrs(int height, int width, int colDist, DPointerArray arrays) {
        this.height = height;
        this.width = width;
        this.colDist = colDist;
        this.arrays = arrays;
    }


    /**
     * Creates a batch of matrices from the sub matrices of contains.
     *
     * @param contains The Matrix containing all the sub matrices that will go
     * into the batch.
     * @param step Accepts a pair of indices in the matrix and moves that pair
     * to the upper left corner of the next matrix to appear in the batch.
     * @param height The height of each sub matrix.
     * @param width The width of each sub matrix.
     * @param batchSize The number of sub matrices.
     */
    public MatricesPntrs(Matrix contains, Consumer<Point> step, int height, int width, int batchSize) {

        this(
                height,
                width,
                contains.colDist,
                getDarray(contains, step, height, width, batchSize)
        );
    }

    /**
     * Creates a batch of matrices from the sub matrices of contains.
     *
     * @param contains The Matrix containing all the sub matrices that will go
     * into the batch.
     * @param downStride How far down to go for the next matrix.
     * @param rightStride Once a column is complete, go this is the distance to
     * the next column.
     * @param height The height of each sub matrix.
     * @param width The width of each sub matrix.
     */
    public MatricesPntrs(Matrix contains, int downStride, int rightStride, int height, int width) {

        this(height, width, height, pointerSeter(contains, downStride,
                rightStride, height, width));
    }

    /**
     * A consumer for setting pointers from a containing matrix.
     *
     * @param contains A matrix containing all the elements to be pointed to.
     * @param downStride How far down each new pointer points.
     * @param rightStride How far to the right each new pointer points.
     * @param height The height of each matrix being pointed to.
     * @param width The width of each matrix being pointed to.
     * @return
     */
    public static DPointerArray pointerSeter(Matrix contains, int downStride, int rightStride, int height, int width) {
        return getDarray(contains, p -> {
            p.y += downStride;
            if (p.y >= contains.getHeight()) {
                p.y = 0;
                p.x += rightStride;
            }
        }, height, width,
                (contains.getHeight() / downStride) * (contains.getWidth() / rightStride));
    }

    /**
     * Extracts a DArray2d to serve as the underlying data for a matrixBatch.
     *
     * @param contains The Matrix containing all the sub matrices that will go
     * into the batch.
     * @param step Accepts a pair of indices in the matrix and moves that pair
     * to the upper left corner of the next matrix to appear in the batch.
     * @param height The height of each sub matrix.
     * @param width The width of each sub matrix.
     * @param batchSize The number of sub matrices.
     * @return The data for each sub matrix.
     */
    public static DPointerArray getDarray(Matrix contains, Consumer<Point> step, int height, int width, int batchSize) {
        DArray[] arrays = new DArray[batchSize];

        Point p = new Point(0, 0);

        for (int i = 0; i < batchSize; i++, step.accept(p))
            arrays[i] = contains.getSubMatrix(p.y, p.y + height, p.x,
                    p.x + width).array();

        return new DPointerArray(contains.getHandle(), arrays);
    }

    /**
     * Adds the result of a matrix-matrix multiplication to this matrix batch.
     *
     * <p>
     * The multiplication is computed as:
     * <pre>
     * this = timesAB * (A * B) + timesThis * this
     * </pre>
     * </p>
     *
     * @param handle The cuBLAS handle used to manage GPU operations.
     * @param timesAB The scalar value to multiply the result of the
     * multiplication of A and B.
     * @param a The left-hand side matrix batch A.
     * @param b The right-hand side matrix batch B.
     * @param timesThis The scalar value to multiply this matrix batch before
     * adding the result.
     * @return This matrix batch (modified in place).
     * @throws DimensionMismatchException If the dimensions of matrix batch A
     * and matrix batch B are not compatible for multiplication.
     * @throws ArrayIndexOutOfBoundsException If the sizes of the matrix batches
     * A, B, and this batch do not match.
     */
    public MatricesPntrs addProduct(Handle handle, double timesAB, MatricesPntrs a, MatricesPntrs b, double timesThis) {
        // Ensure the dimensions are compatible for matrix multiplication
        if (a.width != b.height) {
            throw new DimensionMismatchException(a.width, b.height);
        }
        // Ensure all batches have the same number of matrices
        if (a.arrays.length != b.arrays.length || a.arrays.length != arrays.length) {
            throw new ArrayIndexOutOfBoundsException(
                    "Batches are not the same size. "
                    + "A batch = " + a.arrays.length + ", B batch = " + b.arrays.length
                    + ", this batch = " + arrays.length);
        }

        // Perform the batched matrix-matrix multiplication
        arrays.addProductBatched(handle,
                a.transposeForOperations, b.transposeForOperations,
                a.height, a.width, b.width,
                timesAB,
                a.arrays, a.colDist,
                b.arrays, b.colDist,
                timesThis, colDist,
                arrays.length
        );
        return this;
    }

    /**
     * Returns the height (number of rows) of each matrix in the batch.
     *
     * @return The height of each matrix in the batch.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the width (number of columns) of each matrix in the batch.
     *
     * @return The width of each matrix in the batch.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Returns the column distance (leading dimension) of each matrix in the
     * batch.
     *
     * @return The column distance of each matrix in the batch.
     */
    public int getColDist() {
        return colDist;
    }

    /**
     * Returns the number of matrices in the batch.
     *
     * @return The number of matrices in the batch.
     */
    public int getBatchSize() {
        return arrays.length;
    }

    /**
     * Checks if the batch of matrices is empty.
     *
     * @return True if the batch contains no matrices, false otherwise.
     */
    public boolean isEmpty() {
        return arrays.length == 0;
    }

    /**
     * Transposes each matrix in the batch for future operations.
     *
     * <p>
     * This method sets the {@code transposeForOperations} flag to true, which
     * indicates that matrices should be transposed when used in future
     * operations.
     * </p>
     */
    public void transpose() {
        transposeForOperations = !transposeForOperations;
    }

    /**
     * Resets the transposition flag, so matrices are not transposed in future
     * operations.
     */
    public void resetTranspose() {
        this.transposeForOperations = false;
    }

    /**
     * Copies this batch but still references same location in gpu memory.
     *
     * @return A shallow copy of this batch.
     */
    public MatricesPntrs shallowCopy() {
        MatricesPntrs copy = new MatricesPntrs(height, width, colDist,
                arrays);
        copy.transposeForOperations = transposeForOperations;
        return copy;
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
     * @param fill The part of the dense matrix that is looked at and replaced.
     * @param info infoArray is an integer array of size batchsize. If
     * potrfBatched returns CUSOLVER_STATUS_INVALID_VALUE, infoArray[0] = -i
     * (less than zero), meaning that the i-th parameter is wrong (not counting
     * handle). If potrfBatched returns CUSOLVER_STATUS_SUCCESS but infoArray[i]
     * = k is positive, then i-th matrix is not positive definite and the
     * Cholesky factorization failed at row k.
     */
    public void choleskyFactorization(Handle handle, IArray info, DPointerArray.Fill fill) {
        arrays.choleskyFactorization(handle, width, colDist, info, fill);
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
     * @param b The right hand side of Ax = b that will hold the solution when
     * done.
     * @param fill Indicates whether the upper or lower triangular part of A is
     * stored. It should be either {@link Fill#UPPER} or {@link Fill#LOWER}.
     * @param info An optional output array to store the status of each system
     * in the batch. If `info == null`, an array will be created internally. If
     * info is not null, it must have a length equal to the number of matrices
     * in the batch.
     *
     * @throws IllegalArgumentException if the handle, fillA, or b is null.
     * @throws IllegalArgumentException if any of the dimensions (heightA,
     * widthBAndX, lda, ldb) are not positive.
     */
    public void solveSymmetric(Handle handle, MatricesPntrs b, IArray info, DPointerArray.Fill fill) {
        arrays.solveCholesky(handle, fill, height, colDist, b.arrays, b.colDist,
                info);
    }

    /**
     * Shifts where all the pointer point to.
     *
     * @param handle The handle
     * @param right The distance to shift them to the right.
     * @param down The distance to shift them down.
     */
    public void shiftPointers(Handle handle, int right, int down) {
        Kernel.run("pointerShift",
                handle,
                arrays.length,
                arrays, P.to(1),
                P.to(arrays), 
                P.to(1),
                P.to(down + right * colDist)                
        );
    }

    /**
     * closes the underlying array of pointers to data. This should only be
     * called if the resource is not being used by any other objects. This
     * method does not close the data being pointed to by the elements of the
     * array of pointers.
     */
    @Override
    public void close() {
        arrays.close();
    }


    /**
     * Replaces this matrix with the lu factorization. L has unit diagonal so
     * the diagoanal in this matrix will be U's.
     *
     * @see https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrfbatched
     * @param handle The handle
     * @param pivotArray output indicating which pivots took place. This should
     * have width*batchsize elements.
     * @param info Output indicating what if anything went wrong.Should be
     * batchsize in length.
     */
    public void LUFactor(Handle handle, IArray pivotArray, IArray info) {
        arrays.luFactor(handle, width, colDist, pivotArray, info);
    }

    /**
     * Solves a set of matrix Ax=b where A has undergone LU factorization.
     *
     * @param handle The handle.
     * @param b The solution and right hand side.
     * @param pivotArray The pivot array generated by LU factorization.
     * @param info Information about the result we be placed here.
     */
    public void solveLUFactored(Handle handle, MatricesPntrs b, IArray pivotArray, IArray info) {
        arrays.solveWithLUFactored(
                handle, transposeForOperations,
                width, b.width, colDist, b.colDist,
                pivotArray, b.arrays,
                info
        );
    }

}
