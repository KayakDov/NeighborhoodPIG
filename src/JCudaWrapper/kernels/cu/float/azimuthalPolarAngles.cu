/**
 * CUDA kernel to compute azimuthal (theta) and polar (phi) angles from unit vectors in 3D.
 *
 * Given an array of unit vectors, this kernel computes the corresponding azimuthal (θ)
 * and polar (φ) angles for each vector.
 *
 * @param n            Total number of vectors.
 * @param srcVecs      Pointer to the input array of 3D unit vectors, stored in column-major order.
 * @param ldSrc        Leading dimension of the source vector array.
 * @param heightSrc    Number of rows in the source vector array.
 * @param dstAzimuthal Pointer to the output array for storing azimuthal angles (theta).
 * @param AzHeight     Number of rows in the azimuthal output array.
 * @param ldAz         Leading dimension of the azimuthal output array.
 * @param dstPolar     Pointer to the output array for storing polar angles (phi).
 * @param poHeight     Number of rows in the polar output array.
 * @param ldPo         Leading dimension of the polar output array.
 * @param tolerance    Tolerance value for avoiding singularities.
 */
extern "C" __global__
void toSphericalKernel(
    const int n, 
    const float* srcVecs, const int ldSrc, const int heightSrc, 
    float* dstAzimuthal, const int AzHeight, const int ldAz,
    float* dstPolar, const int poHeight, const int ldPo,
    const float tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;    

    const float* vec = srcVecs + (3 * idx / heightSrc * ldSrc) + (3 * (idx % heightSrc));

    dstAzimuthal[(idx / AzHeight) * ldAz + (idx % AzHeight)] = 
        (fabs(vec[0]) + fabs(vec[1]) <= tolerance) ? NAN : atan2(vec[1], vec[0]);

    dstPolar[(idx / poHeight) * ldPo + (idx % poHeight)] = 
        (fabs(vec[0]) + fabs(vec[1]) + fabs(vec[2]) <= tolerance) ? NAN : acos(vec[2]);
}

