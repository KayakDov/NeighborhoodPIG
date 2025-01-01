package JCudaWrapper.array;

import static JCudaWrapper.array.Array.checkNull;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverEigMode;
import jcuda.jcusolver.gesvdjInfo;
import JCudaWrapper.resourceManagement.Handle;
import static JCudaWrapper.array.Array.checkPositive;
import jcuda.runtime.cudaError;
import jcuda.jcusolver.syevjInfo;

/**
 * A class for a batch of consecutive arrays.
 *
 * @author E. Dov Neimand
 */
public class DStrideArray extends DArray {

    public final int stride, batchSize, subArrayLength;

    /**
     * The constructor. Make sure batchSize * strideSize is less than length 
     * @param p A pointer to the first element. 
     * from the first element of one subsequence to the first element of the next.      
     * @param strideSize The number of elements between the first element of each subarray. 
     * @param batchSize The number of strides. @param subArrayLength The length of e
     * @param subArrayLength The length of each sub arrau/
     * @param deallocateOnClose Dealocate gpu memory when this is closed or inaccessible.
     */
    protected DStrideArray(CUdeviceptr p, int strideSize, int subArrayLength, int batchSize, boolean deallocateOnClose) {
        super(p, totalDataLength(strideSize, subArrayLength, batchSize));
        this.stride = strideSize;
        this.subArrayLength = subArrayLength;
        this.batchSize = batchSize;
    }

    /**
     * The number of sub arrays.
     *
     * @return The number of sub arrays.
     */
    /**
     * The number of sub arrays.
     *
     * @return The number of sub arrays.
     */
    public int batchCount() {
        return batchSize;
    }

    /* Doesn't work because Jacobiparms doesn't work.
     * 
     * Creates an auxiliary workspace for cusolverDnDsyevjBatched using
     * cusolverDnDsyevjBatched_bufferSize.
     *
     * @param handle The cusolverDn handle.
     * @param height The size of the matrices (nxn).
     * @param input The device pointer to the input matrices.
     * @param ldInput The leading dimension of the matrix A.
     * @param resultValues The device pointer to the eigenvalue array.
     * @param batchSize The number of matrices in the batch.
     * @param fill How is the matrix stored.
     * @param params The syevjInfo_t structure for additional parameters.
     * @return A Pointer array where the first element is the workspace size,
     * and the second element is the device pointer to the workspace.
     */
    public int eigenWorkspaceSize(Handle handle,
            int height,
            int ldInput,
            DArray resultValues,
            syevjInfo params,
            DPointerArray.Fill fill) {
        int[] lwork = new int[1];

        JCusolverDn.cusolverDnDsyevjBatched_bufferSize(
                handle.solverHandle(),
                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR,
                fill.getFillMode(),
                height,
                pointer,
                ldInput,
                resultValues.pointer,
                lwork,
                params,
                batchCount()
        );

        return lwork[0];
    }

//    /**
//     * If each sub array is square matrix data,then this is the height and width
//     * of the matrix.
//     *
//     * @return If each sub array is square matrix data,then this is the height
//     * and width of the matrix.
//     */
//    private int squareMatrixHeightWidth() {
//        return (int) Math.round(Math.sqrt(subArrayLength));
//    }
//
////Doesn't work because JacobiParams doesn't work.
//    /**
//     * https://docs.nvidia.com/cuda/cusolver/index.html?highlight=cusolverDnCheevjBatched#cuSolverDN-lt-t-gt-syevjbatch
//     *
//     * Computes the eigenvalues and eigenvectors of a batch of symmetric
//     * matrices using the cuSolver library.
//     *
//     * This method leverages the cusolverDnDsyevjBatched function, which
//     * computes the eigenvalues and eigenvectors of symmetric matrices using the
//     * Jacobi method.
//     *
//     *
//     * This, the input matrices, which must be stored in GPU memory
//     * consecutively so that each matrix is column-major with leading dimension
//     * lda, so the formula for random access is a_k[i, j] = A[i + lda*j +
//     * lda*n*k]
//     *
//     * The input matrices are replaced with the eigenvectors.
//     *
//     * Use createEigenWorkspace to calculate the size of the workspace.
//     *
//     * @param handle
//     * @param ldInput The leading dimension of the input matrices.
//     * @param resultValues Array to store the eigenvalues of the matrices.
//     * @param workSpace An auxilery workspace.
//     * @param cublasFillMode Fill mode for the symmetric matrix (upper or
//     * lower).
//     * @param jp a recourse needed by this method.
//     * @param info An integer array. It's length should be batch count. It
//     * stores error messages.
//     */
//    public void computeEigen(Handle handle,
//            int ldInput, DArray resultValues,
//            DArray workSpace, DPointerArray.Fill cublasFillMode,
//            MySyevjInfo jp, IArray info) {
//
//        JCusolverDn.cusolverDnDsyevjBatched(handle.solverHandle(), // Handle to the cuSolver context
//                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR, // Compute both eigenvalues and eigenvectors
//                cublasFillMode.getFillMode(), // Indicates matrix is symmetric
//                squareMatrixHeightWidth(), pointer, ldInput,
//                resultValues.pointer,
//                workSpace.pointer, workSpace.length,
//                info.pointer, // Array to store status info for each matrix
//                jp.getParams(), // Jacobi algorithm parameters
//                batchCount()
//        );
////            // Step 5: Check for convergence status in info array
////            int[] infoHost = new int[batchCount]; // Host array to fetch status
////
////            try (Handle hand = new Handle()) {
////                info.get(hand, infoHost, 0, 0, batchCount);
////            }
////
////            for (int i = 0; i < batchCount; i++) {
////                if (infoHost[i] != 0) {
////                    System.err.println("Matrix " + i + " failed to converge: info = " + infoHost[i]);
////                }
////            }
//
//    }

    /**
     * Performs batched matrix-matrix multiplication:
     *
     * <pre>
     * Result[i] = alpha * op(A[i]) * op(B[i]) + timesResult * Result[i]
     * </pre>
     *
     * Where op(A) and op(B) can be A and B or their transposes.
     *
     * This method computes multiple matrix-matrix multiplications at once,
     * using strided data access, allowing for efficient batch processing.
     *
     * @param handle Handle to the cuBLAS library context.
     * @param transA True if matrix A should be transposed, false otherwise.
     * @param transB True if matrix B should be transposed, false otherwise.
     * @param aRows The number of rows in matrix A.
     * @param aColsBRows The number of columns in matrix A and the number of
     * rows in matrix B.
     * @param bCols The number of columns in matrix B.
     * @param timesAB Scalar multiplier applied to the matrix-matrix product.
     * @param matA Pointer to the batched matrix A in GPU memory.
     * @param lda Leading dimension of matrix A (the number of elements between
     * consecutive columns in memory).
     * @param matB Pointer to the batched matrix B in GPU memory.
     * @param ldb Leading dimension of matrix B (the number of elements between
     * consecutive columns in memory).
     * @param timesResult Scalar multiplier applied to each result matrix before
     * adding the matrix-matrix product.
     * @param ldResult Leading dimension of the result matrix (the number of
     * elements between consecutive columns in memory).
     *
     */
    public void addProduct(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DStrideArray matA,
            int lda, DStrideArray matB, int ldb, double timesResult,
            int ldResult) {

        checkNull(handle, matA, matB);
        checkPositive(aRows, bCols, ldb, ldResult);
        checkAgainstLength(aRows * bCols * batchCount() - 1);

        int result = JCublas2.cublasDgemmStridedBatched(handle.get(),
                DArray.transpose(transA), DArray.transpose(transB),
                aRows, bCols, aColsBRows,
                P.to(timesAB),
                matA.pointer, lda, matA.stride,
                matB.pointer, ldb, matB.stride,
                P.to(timesResult), pointer, ldResult, stride,
                batchCount()
        );
        if(result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda multiplication failed.");
        
    }

    /**
     * The length of the array.
     *
     * @param batchSize The number of elements in the batch.
     * @param strideSize The distance between the first elements of each batch.
     * @param subArrayLength The length of each subArray.
     * @return The minimum length to hold a batch described by these paramters.
     */
    public static int totalDataLength(int strideSize, int subArrayLength, int batchSize) {
        return strideSize * (batchSize - 1) + subArrayLength;
    }

    /**
     * An empty batch array.
     *
     * @param batchSize The number of subsequences.
     * @param strideSize The size of each subsequence.
     * @param subArrayLength The length of each sub arrays.
     * @return An empty batch array.
     */
    public static DStrideArray empty(int batchSize, int strideSize, int subArrayLength) {

        return new DStrideArray(
                Array.empty(totalDataLength(strideSize, subArrayLength, batchSize), PrimitiveType.DOUBLE),
                strideSize,
                subArrayLength, batchSize, true
        );
    }
    
    /**
     *
     * @param handle
     * @return An array of pointers to each of the subsequences.
     */
    public DPointerArray getPointerArray(Handle handle) {
        return super.getPointerArray(handle, stride);
    }

    /**
     * Performs batched SVD (Singular Value Decomposition) using the
     * cusolverDnSgesvdjBatched function.
     *
     * This function computes the SVD of each matrix in a batch:
     *
     * <pre>
     *     A[i] = U[i] * Sigma[i] * V[i]^T
     * </pre>
     *
     * Each matrix in the batch is decomposed into its left singular vectors
     * (U), singular values (Sigma), and right singular vectors (V^T).
     *
     * @param handle Handle to the cuSOLVER library context.
     * @param computeVectors Specifies whether to compute the full SVD or only
     * the singular values. If true, both U and V^T are computed; otherwise,
     * only singular values.
     * @param rows Number of rows of each matrix.
     * @param cols Number of columns of each matrix.
     * @param lda Leading dimension of each matrix in A (the number of elements
     * between consecutive columns in memory).
     * @param values Pointer to the array of singular values (output).
     * @param leftVectors Pointer to the batched U matrices (output).
     * @param ldleftVecs Leading dimension of each U matrix.
     * @param rightVecs Pointer to the batched V^T matrices (output).
     * @param workSpace
     * @param ldRightVecs Leading dimension of each V matrix.
     * @param info
     * @param batchSize Number of matrices in the batch.
     * @param gesvdj
     *
     */
    public void svdBatched(Handle handle, boolean computeVectors, int rows, int cols, int lda,
            DStrideArray values, DStrideArray leftVectors, int ldleftVecs, DStrideArray rightVecs, int ldRightVecs,
            DArray workSpace, int batchSize, IArray info, gesvdjInfo gesvdj) {

        checkNull(handle, values, leftVectors, rightVecs);
        checkPositive(rows, cols, lda, ldleftVecs, ldRightVecs);
        checkAgainstLength(rows * cols * batchSize - 1);

        int jobzInt = computeVectors ? cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR : cusolverEigMode.CUSOLVER_EIG_MODE_NOVECTOR;

        int error = JCusolverDn.cusolverDnDgesvdjBatched(
                handle.solverHandle(),
                jobzInt, rows, cols,
                pointer, lda,
                values.pointer,
                leftVectors.pointer, ldleftVecs,
                rightVecs.pointer, ldRightVecs,
                workSpace.pointer, workSpace.length,
                info.pointer,
                gesvdj,//I can't seem to create this parameter without a core crash. 
                batchSize
        );
        if (error != cudaError.cudaSuccess)
            throw new RuntimeException("cuda error " + cudaError.stringFor(error));
    }

    /**
     * A sub batch of this batch.
     *
     * @param start The index of the first sub array.
     * @param length The number of sub arrays in this array. Between 0 and batch
     * size.
     * @return A subbatch.
     */
    public DStrideArray subBatch(int start, int length) {
        return subArray(start * stride, totalDataLength(stride, subArrayLength, length)).getAsBatch(stride, subArrayLength, length);
    }
    
    /**
     * Gets the sub array at the given batch index (not to be confused with indices in the underlying array.)
     * @param i The batch index of the desired array: batch index = stride * i     
     * @return The member of the batch at the given batch index.
     */
    public DArray getBatchArray(int i){
        if(i >= batchSize) throw new ArrayIndexOutOfBoundsException();
        return subArray(stride*i, subArrayLength);
    }

    @Override
    public DStrideArray copy(Handle handle) {
        return super.copy(handle).getAsBatch(stride, subArrayLength, batchSize);
    }

    
}
