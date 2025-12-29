/**
 * @class Get
 * @brief A helper class for accessing and modifying values in a batch of 4D data 
 *        (frames × depth × height × width), where each 2D slice (height × width) 
 *        is stored in column-major order.
 *
 * This class computes the appropriate index to retrieve or update a value from 
 * the flattened memory representation using a linear index. It supports accessing 
 * multi-frame volumetric data with strided memory layout.
 */
class Get {
private:
    const int height;     ///< Height of each 2D slice.
    const int idx;        ///< Linear index of the element processed by the thread.
    const int layerSize;  ///< Size of a single 2D slice (height × width).
    const int z;      ///< Index along the depth dimension (z-axis) within a frame.
    const int t;      ///< Index of the t in the 4D dataset (time dimension).
    const int x;
    const int y;
    
public:
    /**
     * @brief Constructs a Get object to calculate indices for accessing a 4D data batch.
     * 
     * @param inputIdx The linear index of the current thread.
     * @param dim      An array containing dimension and stride information, used for index calculation.
     *                   Specifically:
     *                   - dim[0]: Height of each 2D slice.
     *                   - dim[2]: Number of slices along the depth (z) dimension per t.
     *                   - dim[4]: Size of a single 2D slice (height × width).
     *                   - dim[5]: Total size of a frame (depth * height * width).
     */
    __device__ Get(const int idx, const int* dim)
    : idx(idx), 
      height(dim[0]), 
      layerSize(dim[4]),
      z((idx / dim[4]) % dim[2]), 
      t(idx / dim[5]), 
      x(((idx % dim[4]) / dim[0])),
      y(idx % dim[0]) {}

    /**
     * @brief Computes the index into the `dst` pointer array to locate the correct 2D slice.
     * 
     * @param ztLd The leading dimension of the pointer array (`dst`), i.e., the number of slices per frame.
     * @return      The index in `dst` pointing to the appropriate 2D slice.
     */
    __device__ int zt(const int ztLd) const {
        return t * ztLd + z;
    }
    
    /**
     * @brief Computes the column-major index within the current 2D slice.
     * 
     * @param ld   Array of leading dimensions (strides) for each 2D slice.
     * @param ldld Leading dimension (stride) of the ld array across frames.
     * @return     The resolved column-major index for the current linear thread index.
     */
    __device__ int xy(const int* xyLd, const int ldld) const {
        return x * xyLd[zt(ldld)] + y;
    }
    
    /**
     * @brief Retrieves a double value from the source 4D dataset using the calculated indices.
     *
     * @param src   Array of pointers to 2D slices, arranged in t-major and then depth-major order.
     * @param ld    Array of leading dimensions for each slice (used for column-major indexing).
     * @param ldld  Leading dimension of the ld array (stride across layers).
     * @param ztLd Leading dimension of the src array (stride across frames).
     * @return      The double value at the resolved position in the 4D dataset.
     */
    __device__ double val(const double** src, const int* ld, const int ldld, const int ztLd) {
        return src[zt(ztLd)][xy(ld, ldld)];
    }
    
    /**
     * @brief Prints the internal state of the Get object for debugging.
     */
    __device__ void print() const {
        printf("Get(idx = %d, height = %d, layerSize = %d, layer = %d, t = %d, col = %d, row = %d)\n",
               idx, height, layerSize, z, t, y, x);
    }
};

/**
 * @brief CUDA kernel that performs elementwise computation:
 *        `dst = src`
 *
 * This kernel operates on 4D batched data (frames × depth × height × width), allowing
 * for strided memory access via pointer arrays and leading-dimension arrays. It reads
 * corresponding values from the source (`src`), and writes it to the destination (`dst`).
 *
 * @param n         Total number of elements to process (threads).
 * @param dst       Pointer array to output 2D slices (float) (modifiable).
 * @param xyLdDst   Leading dimension array for `dst` (per slice).
 * @param ldldDst   Stride across `xyLdDst` for indexing slices.
 * @param ztLdDst   Stride across `dst` for frame × depth indexing.
 * @param src       Pointer array to input 2D slices (double).
 * @param xyLdSrc   Leading dimension array for `src` (per slice).
 * @param ldldSrc   Stride across `xyLdSrc` for indexing slices.
 * @param ztLdSrc   Stride across `src` for frame × depth indexing.
 * @param dim       An array containing dimension and stride information, used for index calculation in Get class.
 */
extern "C" __global__ void mapToFloatKernel(
    const int n,
    float** dst, const int* xyLdDst, const int ldldDst, const int ztLdDst,
    const double** src, const int* xyLdSrc, const int ldldSrc, const int ztLdSrc,
    const int* dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Get ind(idx, dim);

    dst[ind.zt(ztLdDst)][ind.xy(xyLdDst, ldldDst)] = (float)ind.val(src, xyLdSrc, ldldSrc, ztLdSrc);
}

