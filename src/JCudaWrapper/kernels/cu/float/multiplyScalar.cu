/**
 * CUDA kernel for element-wise multiplication of a matrix by a scalar.
 * 
 * Each thread computes one element of the destination matrix.
 * The kernel assumes column-major storage and allows different leading dimensions (ld) for source and destination matrices.
 * 
 * @param n         The total number of elements to process.
 * @param dst       Pointer to the destination matrix in device memory.
 * @param ldDst     Leading dimension (stride) of the destination matrix.
 * @param heightDst The number of rows in the destination matrix.
 * @param src       Pointer to the source matrix in device memory.
 * @param ldSource  Leading dimension (stride) of the source matrix.
 * @param heightSrc The number of rows in the source matrix.
 * @param scalar    The scalar value to multiply each element by.
 */
extern "C" __global__ void multiplyScalarKernel(
    const int n,
    float* dst,
    const int ldDst,
    const int heightDst,
    const float* src,
    const int ldSrc,
    const int heightSrc,
    const float scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
      
    dst[(idx / heightDst) * ldDst + idx % heightDst] = scalar * src[(idx / heightSrc) * ldSrc + idx % heightSrc];
}

