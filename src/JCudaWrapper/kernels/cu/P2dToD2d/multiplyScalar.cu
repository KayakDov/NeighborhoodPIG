/**
 * @class IndexMapper
 * @brief Maps a flattened thread index to 4D data indices (frame × depth × height × width),
 *        supporting column-major storage and strided memory layouts.
 */
class IndexMapper {
private:
    const int height;   ///< Height (rows) of each 2D layer.
    const int idx;     ///< Flattened thread/global index.
    const int layerArea;     ///< Number of elements in a single 2D layer (height × width).
    const int layerInd;    ///< Index along the depth (z-axis) within a frame.
    const int frameInd;    ///< Index along the time (frame) axis.

public:
    /**
     * @brief Constructor that maps a flat index to 4D coordinates.
     *
     * @param globalIndex Global linear index across all 4D elements.
     * @param dim  The dimensions.
     */
    __device__ IndexMapper(const int globalIndex, const int* dim)
        : idx(globalIndex),
          height(dim[0]),
          layerArea(dim[4]),
          layerInd((globalIndex / layerArea) % dim[2]),
          frameInd(globalIndex / dim[5]) {}

    /**
     * @brief Multiplies the destination value by a scalar in-place.
     *
     * @param pointersToLayers 2D array of pointers to 2D layers (frames × depth).
     * @param ldLayers         2D array of column-major strides for each 2D layer.
     * @param ldld             Leading dimension of ldLayers (stride across depth).
     * @param ldPtrs           Leading dimension of pointersToLayers (stride across frames).
     * @param scalar           Scalar multiplier.
     */
    __device__ void multiply(float** pointersToLayers, const int* ldLayers, const int ldld, const int ldPtrs, double scalar) {
        pointersToLayers[ptrIndex(ldPtrs)][elementIndex(ldLayers, ldld)] *= scalar;
    }

    /**
     * @brief Computes column-major element index in the layer.
     *
     * @param ldLayers 2D array of per-layer column strides.
     * @param ldld     Leading dimension of ldLayers.
     * @return         Offset into the 2D layer data.
     */
    __device__ int elementIndex(const int* ldLayers, const int ldld) const {
    }

    /**
     * @brief Computes the index into pointersToLayers.
     *
     * @param ldPtrs Leading dimension (stride) across frames in pointersToLayers.
     * @return       Index pointing to the correct 2D layer pointer.
     */
    __device__ int ptrIndex(const int ldPtrs) const {
        return frameInd * ldPtrs + layerInd;
    }
    
    /**
     * @brief Prints the current index mapping for debugging.
     */
    __device__ void print() const {
        printf("Global Index: %d, Frame: %d, Layer: %d, Row: %d, Col: %d, Slice Area: %d\n",
               idx, frameInd, layerInd, idx % height, idx / height, layerArea);
    }
};


/**
 * @brief CUDA kernel that multiplies each element in a 4D dataset by a scalar.
 *
 * The dataset is organized as: frames × depth × height × width. Each 2D layer
 * is stored in column-major order and accessed via pointers with stride metadata.
 *
 * @param totalElements   Total number of elements to process.
 * @param pointersToLayers 2D array of pointers to 2D layers.
 * @param ldLayers        2D array of column strides per layer.
 * @param ldld            Leading dimension of ldLayers (stride across depth).
 * @param ldPtrs          Leading dimension of pointersToLayers (stride across frames).
 * @param dim The dimensions
 * @param scalar          Scalar to multiply each element by.
 */
extern "C" __global__ void multiplyScalarKernel(
    const int totalElements,
    float** pointersToLayers, const int* ldLayers, const int ldld, const int ldPtrs,
    const int* dim,
    const double scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    IndexMapper mapper(idx, dim);
        
    mapper.multiply(pointersToLayers, ldLayers, ldld, ldPtrs, scalar);
}

