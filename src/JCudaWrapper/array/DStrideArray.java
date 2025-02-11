package JCudaWrapper.array;

import jcuda.jcublas.JCublas2;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.runtime.cudaError;

/**
 * A class for a batch of consecutive arrays.
 *
 * @author E. Dov Neimand
 */
public interface DStrideArray extends StrideArray, DArray{



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
    public default void addProduct(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DStrideArray matA,
            int lda, DStrideArray matB, int ldb, double timesResult,
            int ldResult) {

        int result = JCublas2.cublasDgemmStridedBatched(handle.get(),
                Array.transpose(transA), Array.transpose(transB),
                aRows, bCols, aColsBRows,
                P.to(timesAB),
                matA.pointer(), lda, matA.stride(),
                matB.pointer(), ldb, matB.stride(),
                P.to(timesResult), pointer(), ldResult, stride(),
                batchSize()
        );
        if(result != cudaError.cudaSuccess)
            throw new RuntimeException("cuda multiplication failed.");
        
    }
    

    
    /**
     * Gets the sub array at the given batch index (not to be confused with indices in the underlying array.)
     * @param i The batch index of the desired array: batch index = stride * i     
     * @return The member of the batch at the given batch index.
     */
    public default DArray getBatchArray(int i){

        return new DArray1d(this, stride()*i, subArraySize());
    }   
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
//
//    /* Doesn't work because Jacobiparms doesn't work.
//     * 
//     * Creates an auxiliary workspace for cusolverDnDsyevjBatched using
//     * cusolverDnDsyevjBatched_bufferSize.
//     *
//     * @param handle The cusolverDn handle.
//     * @param height The size of the matrices (nxn).
//     * @param input The device pointer to the input matrices.
//     * @param ldInput The leading dimension of the matrix A.
//     * @param resultValues The device pointer to the eigenvalue array.
//     * @param batchSize The number of matrices in the batch.
//     * @param fill How is the matrix stored.
//     * @param params The syevjInfo_t structure for additional parameters.
//     * @return A Pointer array where the first element is the workspace size,
//     * and the second element is the device pointer to the workspace.
//     */
//    public int eigenWorkspaceSize(Handle handle,
//            int height,
//            int ldInput,
//            DArray3d resultValues,
//            syevjInfo params,
//            DPointerArray.Fill fill) {
//        int[] lwork = new int[1];
//
//        JCusolverDn.cusolverDnDsyevjBatched_bufferSize(
//                handle.solverHandle(),
//                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR,
//                fill.getFillMode(),
//                height,
//                pointer,
//                ldInput,
//                resultValues.pointer,
//                lwork,
//                params,
//                batchCount()
//        );
//
//        return lwork[0];
//    }

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
//    public void svdBatched(Handle handle, boolean computeVectors, int rows, int cols, int lda,
//            DStrideArray values, DStrideArray leftVectors, int ldleftVecs, DStrideArray rightVecs, int ldRightVecs,
//            DArray workSpace, int batchSize, IArray info, gesvdjInfo gesvdj) {
//
//        int jobzInt = computeVectors ? cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR : cusolverEigMode.CUSOLVER_EIG_MODE_NOVECTOR;
//
//        opCheck(JCusolverDn.cusolverDnDgesvdjBatched(
//                handle.solverHandle(),
//                jobzInt, rows, cols,
//                pointer(), lda,
//                values.pointer(),
//                leftVectors.pointer(), ldleftVecs,
//                rightVecs.pointer(), ldRightVecs,
//                workSpace.pointer(), workSpace.size(),
//                info.pointer(),
//                gesvdj,//I can't seem to create this parameter without a core crash. 
//                batchSize
//        ));
//    }