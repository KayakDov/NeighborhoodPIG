
/**
 * @brief Computes the coherence measure for a batch of eigenvalues.
 *
 * The coherence measure is calculated as the normalized difference between 
 * the largest and second-largest eigenvalues in the input, optionally accounting 
 * for 3D data. The results are written to the output array.
 *
 * @param n        The total number of data points to process.
 * @param eigenVals Pointer to the flattened array of eigenvalues for all data points.
 *                  Eigenvalues for each data point are stored sequentially.
 * @param mapTo    Pointer to the array where coherence results will be stored.
 * @param is3d     A flag indicating whether to consider the third eigenvalue (true for 3D data).
 * @param tolerance A small value used to check for near-zero denominators to avoid division errors.
 *
 * @details 
 * Each thread processes a single data point. For 2D data, only the first two 
 * eigenvalues are considered, while for 3D data, all three eigenvalues are used.
 * The coherence measure is calculated as:
 * 
 * \f[
 * result = \frac{e_0 - e_1}{\text{denom}}
 * \f]
 * 
 * where \f$ e_0, e_1, e_2 \f$ are the eigenvalues (sorted in descending order),
 * and \f$ \text{denom} \f$ is the sum of the eigenvalues (up to two or three, 
 * depending on `is3d`).
 *
 * If the denominator is smaller than \f$ 3 \times \text{tolerance} \f$, the result is set to 0.
 *
 * @note This kernel assumes that the eigenvalues are pre-sorted in descending order.
 * 
 * @warning The value of `tolerance` should be chosen carefully to avoid numerical instability.
 *
 * @example
 * // Example usage in host code
 * coherenceKernel<<<numBlocks, blockSize>>>(n, eigenVals, mapTo, is3d, 1e-6);
 */
extern "C" __global__ void coherenceKernel(
    const int n, 
    const double* eigenVals, 
    const int ldFrom, 
    double* mapTo, 
    const bool is3d, 
    const double tolerance
) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx >= n) return;
     
    const double* e = eigenVals + idx * ldFrom; 
    
    double& result = *(mapTo + idx); 
    
    if (e[0] <=  tolerance) result = 0;
    
    else result = (e[0] - e[1]) / (e[0] + e[1] + (is3d? e[2]:0));
}

