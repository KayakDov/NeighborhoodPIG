#include <cmath>  // Required for nan("")

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
 * @param dstPolar     Pointer to the output array for storing polar angles (phi).
 * @param poHeight     Number of rows in the polar output array.
 * @param ldPo         Leading dimension of the polar output array.
 * @param tolerance    Tolerance value for avoiding singularities.
 */
extern "C" __global__
void toSphericalKernel(
    const int n, 
    const double* srcVecs, const int heightSrc, const int ldSrc,
    double* dstAzimuthal, const int azHeight, const int ldAz,
    double* dstPolar, const int poHeight, const int ldPo,
    const double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;    

    const double* vec = srcVecs + 3 * idx / heightSrc * ldSrc + 3 * idx % heightSrc;

    dstAzimuthal[(idx / azHeight) * ldAz + (idx % azHeight)] = 
        (fabs(vec[0]) <= tolerance && fabs(vec[1]) <= tolerance) ? nan("") : atan2(vec[1], vec[0]);

	int polarInd = (idx / poHeight) * ldPo + idx % poHeight;
	if(vec[2] > 1 - tolerance) dstPolar[polarInd] = 0;
	else if(vec[2] < tolerance - 1) dstPolar[polarInd] = 3.14159265;
	else if(fabs(vec[2]) + fabs(vec[1]) + fabs(vec[0]) < tolerance ) dstPolar[polarInd] = nan("");	
	else dstPolar[polarInd] = acos(vec[2]);
	
}

