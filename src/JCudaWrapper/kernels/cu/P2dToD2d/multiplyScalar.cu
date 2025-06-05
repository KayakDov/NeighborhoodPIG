
/**
 * @class Get
 * @brief A helper class for accessing values within a batch of 3D data (frames x depth x height x width),
 * assuming each 2D slice (height x width) is stored in column-major order. This class calculates
 * the appropriate index to retrieve a value based on a flattened linear index.
 */
class Get{
public:
    const int idx;             ///< Linear index of the element being processed by the current thread.    
    const int layerSize;       ///< Size of a single 2D slice (height * width).
    const int layer;           ///< Index of the current slice along the depth dimension (0 to depth - 1).
    const int frame;           ///< Index of the current frame.
    const int row;
    const int col;

    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     * @param downSampleFactorXY The downsampling factor applied in the x and y dimensions.
     */
    __device__ Get(const int idx, const int* dim)
    : idx(idx), 
      layerSize(dim[4]), 
      layer((idx / dim[4]) % dim[2]), 
      frame(idx / dim[5]),
      row(idx % dim[0]),
      col((idx % dim[4])/dim[0]) {}

    

    /**
     * @brief Computes the column-major index within a single 2D slice (height x width).
     * @tparam T The data type of the array elements.
     * @param ld Array of leading dimensions for each 2D slice.
     * @param ldld Leading dimension of the ld array.
     * @return The column-major index within the current 2D slice.
     */
    __device__ int word(const int* ld, const int ldld) const{
        return col * ld[page(ldld)] + row;
    }

    /**
     * @brief Computes the index into the array of pointers (`src`) to access the correct 2D slice.
     * @param ldPtr Leading dimension of the array of pointers.
     * @return The index of the pointer to the current 2D slice.
     */
    __device__ int page(const int ldPtr) const{
        return frame * ldPtr + layer;
    }
    
    
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
     template <typename T>
    __device__ T operator()(const T** src, const int* ld, const int ldld, const int ldPtr) {
        return src[page(ldPtr)][word(ld, ldld)];
    }
    
    template <typename T>
    __device__ T operator()(T** src, const int* ld, const int ldld, const int ldPtr) {
        return src[page(ldPtr)][word(ld, ldld)];
    }
    
    
    /**
     * @brief Sets the value from the source data array based on the calculated multi-dimensional index.
     * @tparam T The data type of the array elements.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
     template <typename T>
    __device__ void multiply(T** src, const int* ld, const int ldld, const int ldPtr, T val) {
        src[page(ldPtr)][word(ld, ldld)] *= val;
    }
    
    /**
     * @brief Prints the internal state of the Get object.
     */
    __device__ void print(const int* ld, int ldld, int ldPtr) const {
        printf("Get\n idx: %d, frame: %d, layer: %d, lyayerSize: %d, \ncol: %d, row: %d, page: %d, word: %d, ld: %d, ldld: %d, ldPtr: %d\n\n",
               idx, frame, layer, layerSize, col, row, page(ldPtr), word(ld, ldld), ld[page(ldld)], ldld, ldPtr);
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
    const float scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    Get mapper(idx, dim);

    mapper.multiply(pointersToLayers, ldLayers, ldld, ldPtrs, scalar);
}

