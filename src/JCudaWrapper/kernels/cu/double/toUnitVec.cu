/**
 * Computes the linear memory index for a 3D vector stored in column-major format.
 * @param idx The flattened index.
 * @param ld The leading dimension (stride between columns).
 * @param height The height of the matrix.
 * @return The computed memory index.
 */
__device__ int ind(int idx, int ld, int height) {
    return 3 * idx / height * ld + 3 * idx % height;
}

/**
 * Kernel function to normalize 3D vectors in a batched matrix.
 * Each vector is transformed into a unit vector by dividing by its magnitude.
 * 
 * @param n Total number of vectors to process.
 * @param srcVecs Pointer to the input vector data.
 * @param heightSrc Height of the source matrix.
 * @param ldSrc Leading dimension of the source matrix.
 * @param dstVecs Pointer to the output vector data.
 * @param heightDst Height of the destination matrix.
 * @param ldDst Leading dimension of the destination matrix.
 * @param tolerance Small value to prevent division by zero.
 */
extern "C" __global__ void toUnitVecKernel(
    const int n, 
    double* srcVecs, const int heightSrc, const int ldSrc,
    double* dstVecs, const int heightDst, const int ldDst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double* src = srcVecs + ind(idx, ldSrc, heightSrc);
    double* dst = dstVecs + ind(idx, ldDst, heightDst);

    double mag = sqrt(src[0] * src[0] + src[1] * src[1] + src[2] * src[2]);

    for (int i = 0; i < 3; i++) dst[i] = src[i] / mag;
}

