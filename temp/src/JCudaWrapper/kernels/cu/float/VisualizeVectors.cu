/**
 * @file visualizeVectorsKernel.cu
 * @brief CUDA kernel for visualizing vectors.
 */

#include <cuda_runtime.h>

/**
 * @brief Visualizes vectors onto an output image.
 *
 * This CUDA kernel takes a set of vectors, their coherence values, and renders them
 * onto an output image. Each vector is visualized as a line segment centered at a
 * specific location in the output image. The color of the line segment is determined
 * by the corresponding coherence value.
 *
 * @param n           Number of vectors to visualize.
 * @param vectors     Pointer to the input vector data (device memory).
 * @param ldVecs      Leading dimension of the vectors array.
 * @param to          Pointer to the output image data (device memory).
 * @param ldTo        Leading dimension of the output image array.
 * @param coherence   Pointer to the coherence values (device memory).
 * @param ldCoh       Leading dimension of the coherence array.
 * @param height      Height of the output image.
 * @param width       Width of the output image.
 * @param rsltSpacing Spacing between visualized vectors in the output image.
 * @param vecLength   Length of the visualized line segment for each vector.
 */
extern "C" __global__ void visualizeVectorsKernel(
    const int n,
    const float *vectors, const int ldVecs,
    float *to, const int ldTo,
    const float *coherence, const int ldCoh,
    const int height, const int width, const int depth,
    const int rsltSpacing, const int vecLength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    int col = idx / height;
    int row = idx % height;
    int layer = idx / (height * width); 

    const float *vec = vectors + col * ldVecs + 3 * row;

    float coh = coherence[col * ldCoh + row];

    int rad = vecLength / 2;

    for (int i = -rad; i <= rad; i++) {
    
	int x = col * rsltSpacing + (int)std::round(i * vec[0]);
	int y = row * rsltSpacing + (int)std::round(i * vec[1]);
	int z = layer * rsltSpacing + (int)std::round(i * vec[2]);	
	
	if(x < width && x >= 0 && y < height && y >= 0 && z < depth && z >= 0)
		to[y + (x + z * width) * ldTo] = coh;
    }
}
