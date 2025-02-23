
/**
 * CUDA kernel to compute atan2(y, x) for each 2D vector in a column-major matrix.
 * 
 * Each thread processes a single vector, computing the angle theta = atan2(y, x).
 * The vectors are assumed to be stored in a column-major matrix.
 *
 * @param n        Total number of vectors (each with 2 components).
 * @param srcVecs  Pointer to input vectors stored in column-major format.
 * @param ldSrc    Leading dimension (stride) of the source matrix.
 * @param dstAng   Pointer to output angles in column-major format.
 * @param dstHeight Number of rows in the destination matrix.
 * @param ldDst    Leading dimension (stride) of the destination matrix.
 */

extern "C" __global__
void atan2Kernel(const int n, const double* srcVecs, const int ldSrc, double* dstAng, const int dstHeight, const int ldDst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;    
    
    const double* vecFrom = srcVecs + idx*ldSrc;
    
    dstAng[(idx / dstHeight) * ldDst + idx % dstHeight] = atan2(vecFrom[1], vecFrom[0]);    
}
