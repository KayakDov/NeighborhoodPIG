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
    const int layerSize;  ///< Size of a single 3D volume (height × width).
    const int layer;      ///< Index along the depth dimension (z-axis) within a frame.
    const int frame;      ///< Index of the frame in the 4D dataset (time dimension).
    
public:
    /**
     * @brief Constructs a Get object to calculate indices for accessing a 4D data batch.
     * 
     * @param inputIdx The linear index of the current thread.
     * @param height   The height (number of rows) of each 2D slice.
     * @param width    The width (number of columns) of each 2D slice.
     * @param depth    The number of slices along the depth (z) dimension per frame.
     */
    __device__ Get(const int inputIdx, const int height, const int width, const int depth)
    : idx(inputIdx), height(height), layerSize(height * width),
      layer((idx / layerSize) % depth), frame(idx / (layerSize * depth)) {}

    /**
     * @brief Retrieves a value from the source 4D dataset using the calculated indices.
     *
     * @param src   Array of pointers to 2D slices, arranged in frame-major and then depth-major order.
     * @param ld    Array of leading dimensions for each slice (used for column-major indexing).
     * @param ldld  Leading dimension of the ld array (stride across layers).
     * @param ldPtr Leading dimension of the src array (stride across frames).
     * @return      The double value at the resolved position in the 4D dataset.
     */
    __device__ double operator()(const double** src, const int* ld, const int ldld, const int ldPtr) {
        return src[layerInd(ldPtr)][ind(ld, ldld)];
    }
    
    /**
     * @brief Retrieves a value from the source 4D dataset using the calculated indices.
     *
     * @param src   Array of pointers to 2D slices, arranged in frame-major and then depth-major order.
     * @param ld    Array of leading dimensions for each slice (used for column-major indexing).
     * @param ldld  Leading dimension of the ld array (stride across layers).
     * @param ldPtr Leading dimension of the src array (stride across frames).
     * @return      The double value at the resolved position in the 4D dataset.
     */
    __device__ double operator()(double** src, const int* ld, const int ldld, const int ldPtr) {
	return src[layerInd(ldPtr)][ind(ld, ldld)];
    }

    /**
     * @brief Sets a value in the destination 4D dataset using the calculated indices.
     *
     * @param src   Array of pointers to 2D slices, arranged in frame-major and then depth-major order.
     * @param ld    Array of leading dimensions for each slice (used for column-major indexing).
     * @param ldld  Leading dimension of the ld array (stride across layers).
     * @param ldPtr Leading dimension of the src array (stride across frames).
     * @param val   The double value to store in the specified location.
     */
    __device__ void set(double** src, const int* ld, const int ldld, const int ldPtr, double val) {
        src[layerInd(ldPtr)][ind(ld, ldld)] = val;
    }

    /**
     * @brief Computes the column-major index within the current 2D slice.
     * 
     * @param ld   Array of leading dimensions (strides) for each 2D slice.
     * @param ldld Leading dimension (stride) of the ld array across frames.
     * @return     The resolved column-major index for the current linear thread index.
     */
    __device__ int ind(const int* ld, const int ldld) const {
        return (idx / height) * ld[frame * ldld + layer] + idx % height;
    }

    /**
     * @brief Computes the index into the `src` pointer array to locate the correct 2D slice.
     * 
     * @param ldPtr The leading dimension of the pointer array (`src`), i.e., the number of slices per frame.
     * @return      The index in `src` pointing to the appropriate 2D slice.
     */
    __device__ int layerInd(const int ldPtr) const {
        return frame * ldPtr + layer;
    }
};

/**
 * @brief CUDA kernel that performs elementwise computation:
 *        `dst = timesDst * dst + timesProduct * a * b`
 *
 * This kernel operates on 4D batched data (frames × depth × height × width), allowing
 * for strided memory access via pointer arrays and leading-dimension arrays. It reads
 * corresponding values from two inputs (`a` and `b`), multiplies them, scales the result,
 * and accumulates it with a scaled value from the destination (`dst`).
 *
 * @param n         Total number of elements to process (threads).
 * @param dst       Pointer array to output 2D slices (modifiable).
 * @param xyLdDst   Leading dimension array for `dst` (per slice).
 * @param ldldDst   Stride across `xyLdDst` for indexing slices.
 * @param ztLdDst   Stride across `dst` for frame × depth indexing.
 * 
 * @param height    Height of each 2D slice.
 * @param width     Width of each 2D slice.
 * @param depth     Number of slices (depth) per frame.
 * 
 * @param timesProduct Scalar multiplier for the product of `a` and `b`.
 * 
 * @param a         Pointer array to 2D slices of input A.
 * @param xyLdA     Leading dimension array for input A slices.
 * @param ldldA     Stride across `xyLdA` for indexing.
 * @param ztLdA     Stride across `a` for frame × depth indexing.
 * 
 * @param b         Pointer to 1D flattened array of input B.
 * @param xyLdB     Leading dimension array for input B.
 * @param ldldB     Stride across `xyLdB` for indexing.
 * @param ztLdB     Stride across `b` for frame × depth indexing.
 * 
 * @param timesDst  Scalar multiplier applied to the existing value in `dst`.
 */
extern "C" __global__ void addEBEProductKernel(
    const int n,
    double** dst, const int* xyLdDst, const int ldldDst, const int ztLdDst,
    
    const int height, const int width, const int depth,
    
    const double timesProduct,
    
    const double** a, const int* xyLdA, const int ldldA, const int ztLdA,
    
    const double** b, const int* xyLdB, const int ldldB, const int ztLdB,
    
    const double timesDst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Get ind(idx, height, width, depth);

    ind.set(
        dst,
        xyLdDst,
        ldldDst,
        ztLdDst,
        timesDst * ind(dst, xyLdDst, ldldDst, ztLdDst) + timesProduct * ind(a, xyLdA, ldldA, ztLdA) * ind(b, xyLdB, ldldB, ztLdB)
    );
}

