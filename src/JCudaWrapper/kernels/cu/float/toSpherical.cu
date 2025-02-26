#include <cmath>  // Required for nan("")

/**
 * Computes the linear index in a column-major order matrix.
 *
 * @param idx The 1D index in a flattened array.
 * @param ld The leading dimension (stride between columns in memory).
 * @param height The number of rows in the matrix.
 * @return The column-major index.
 */
__device__ int ind(int idx, int ld, int height) {
    return (idx / height) * ld + (idx % height);
}

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
 * @param azHeight     Number of rows in the azimuthal output array.
 * @param ldAz         Leading dimension of the azimuthal output array.
 * @param dstZenith     Pointer to the output array for storing polar angles (phi).
 * @param zeHeight     Number of rows in the polar output array.
 * @param ldZe         Leading dimension of the polar output array.
 * @param tolerance    Tolerance value for avoiding singularities.
 */
extern "C" __global__
void toSphericalKernel(
    const int n, 
    const float* srcVecs, const int heightSrc, const int ldSrc,
    float* dstAzimuthal, const int azHeight, const int ldAz,
    float* dstZenith, const int zeHeight, const int ldZe,
    const float tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;    

    const float* vec = srcVecs + ind(3*idx, ldSrc, heightSrc);

    dstAzimuthal[ind(idx, ldAz, azHeight)] = (fabs(vec[0]) <= tolerance && fabs(vec[1]) <= tolerance) ? nan("") : atan2(vec[1], vec[0]);

	int polarInd = ind(idx,ldZe, zeHeight);	
	
	if(vec[2] > 1 - tolerance) dstZenith[polarInd] = 0;
	else if(vec[2] < tolerance - 1) dstZenith[polarInd] = 3.14159265;
	else if(fabs(vec[2]) + fabs(vec[1]) + fabs(vec[0]) < tolerance) dstZenith[polarInd] = nan("");
	else dstZenith[polarInd] = acos(vec[2]);
}

