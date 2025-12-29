
/**
 * @brief CUDA kernel to compute the element-by-element product of two arrays and store the result in a target array.
 *
 * This kernel calculates the product of corresponding elements in two input arrays `a` and `b`,
 * and writes the result to the output array `to`. The kernel supports arrays with stride increments,
 * allowing it to operate on non-contiguous memory layouts.
 *
 * @param n      The number of elements to process.
 * @param a      Pointer to the first input array.
 * @param ldA   Stride increment for the first input array.
 * @param b      Pointer to the second input array.
 * @param ldB   Stride increment for the second input array.
 * @param dst     Pointer to the output array.
 * @param ldDst  Stride increment for the output array.
 *
 * @note This kernel assumes that the input arrays `a`, `b`, and `to` have been allocated with sufficient size
 * to accommodate the specified strides and number of elements.
 *
 * Make sure to launch the kernel with an appropriate block and grid size configuration.
 */
extern "C" __global__ void addEBEProductKernel(
    const int n,
    float* dst,
    const int ldDst,
    const int heightDst,
    const int heightSrc,
    const float timesProduct,
    const float* a, 
    const int ldA, 
    const float* b, 
    const int ldB,
    const float timesDst
        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    int row = idx % heightSrc;
    int col = idx / heightSrc;
    int indDst = (idx / heightDst) * ldDst + idx % heightDst;  
    dst[indDst] = timesDst * dst[indDst] + timesProduct * a[col * ldA + row] * b[col * ldB + row];
}

