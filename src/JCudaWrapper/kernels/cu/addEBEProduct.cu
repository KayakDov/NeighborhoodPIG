/**
 * @brief CUDA kernel to compute the element-by-element product of two arrays and store the result in a target array.
 *
 * This kernel calculates the product of corresponding elements in two input arrays `a` and `b`,
 * and writes the result to the output array `to`. The kernel supports arrays with stride increments,
 * allowing it to operate on non-contiguous memory layouts.
 *
 * @param n      The number of elements to process.
 * @param a      Pointer to the first input array.
 * @param aInc   Stride increment for the first input array.
 * @param b      Pointer to the second input array.
 * @param bInc   Stride increment for the second input array.
 * @param to     Pointer to the output array.
 * @param toInc  Stride increment for the output array.
 *
 * @note This kernel assumes that the input arrays `a`, `b`, and `to` have been allocated with sufficient size
 * to accommodate the specified strides and number of elements.
 *
 * Example:
 * Given n = 5, aInc = 2, bInc = 2, toInc = 1, this kernel computes:
 * to[0] = a[0] * b[0]
 * to[1] = a[2] * b[2]
 * to[2] = a[4] * b[4]
 * to[3] = a[6] * b[6]
 * to[4] = a[8] * b[8]
 *
 * Make sure to launch the kernel with an appropriate block and grid size configuration.
 */
extern "C" __global__ void addEBEProductKernel(
    const int n,
    double* to,
    const int toInc,
    const double timesProduct,
    const double* a, 
    const int aInc, 
    const double* b, 
    const int bInc,
    const double timesThis
        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    int toInd = idx*toInc;
    to[toInd] = timesThis * to[toInd] + timesProduct * a[idx * aInc] * b[idx * bInc];
}

