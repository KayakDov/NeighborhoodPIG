#include <cmath>  // For std::round
#include <math_constants.h>

/**
 * @class Get
 * @brief A helper class for accessing values within a batch of 3D data (frames x depth x height x width),
 * assuming each 2D slice (height x width) is stored in column-major order. This class calculates
 * the appropriate index to retrieve a value based on a flattened linear index.
 */
class Get{
private:
    const int height;          ///< Height of each 2D slice.
    const int idx;             ///< Linear index of the element being processed by the current thread.  
    const int layerSize;       ///< Size of a single 2D slice (height * width).
    const int layer;           ///< Index of the current slice along the depth dimension (0 to depth - 1).
    const int frame;           ///< Index of the current frame.
public:
    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param height The height of each 2D slice.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     */
    __device__ Get(const int inputIdx, const int height, const int width, const int depth)
    : idx(inputIdx), height(height), layerSize(height * width), layer((idx / layerSize) % depth), frame(idx / (layerSize * depth)) {}

    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
    __device__ double operator()(const double** src, const int* ld, const int ldld, const int ldPtr) const{
        return src[layerInd(ldPtr)][ind(ld, ldld)];
    }
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
    __device__ int operator()(const int** src, const int* ld, const int ldld, const int ldPtr) const{
        return src[layerInd(ldPtr)][ind(ld, ldld)];
    }
    
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
    __device__ void set(double** src, const int* ld, const int ldld, const int ldPtr, const double val) const{
        src[layerInd(ldPtr)][ind(ld, ldld)] = val;
    }
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
    __device__ void set(int** src, const int* ld, const int ldld, const int ldPtr, const int val) const{
        src[layerInd(ldPtr)][ind(ld, ldld)] = val;
    }

    /**
     * @brief Computes the column-major index within a single 2D slice (height x width).
     * @param ld Array of leading dimensions for each 2D slice.
     * @param ldld Leading dimension of the ld array.
     * @return The column-major index within the current 2D slice.
     */
    __device__ int ind(const int* ld, const int ldld) const{
        return (idx / height) * ld[frame * ldld + layer] + idx % height;
    }

    /**
     * @brief Computes the index into the array of pointers (`src`) to access the correct 2D slice.
     * @param ldPtr Leading dimension of the array of pointers.
     * @return The index of the pointer to the current 2D slice.
     */
    __device__ int layerInd(const int ldPtr) const{
        return frame * ldPtr + layer;
    }
};
/**
 * @brief Computes the color intensity for a downward transition in the RGB spectrum.
 * 
 * @param a The base RGB component index (0-5) within the color cycle.
 * @param theta The angle in radians.
 * @return The calculated intensity value (0-255) for the downward transition.
 */
__device__ double down(int a, double theta) {
    return 255 * (1 + a - 3 * theta / CUDART_PI);
}

/**
 * @brief Computes the color intensity for an upward transition in the RGB spectrum.
 * 
 * @param a The base RGB component index (0-5) within the color cycle.
 * @param theta The angle in radians.
 * @return The calculated intensity value (0-255) for the upward transition.
 */
__device__ double up(int a, double theta) {
    return 255 * (3 * theta / CUDART_PI - a);
}

/**
 * @class Crayon
 * @brief Utility class for coloring pixels using angle-based RGB encoding.
 */
class Crayon{
private:
    const Get& at;           ///< Accessor object to determine pixel location.
    int** colors;            ///< Output color array.
    const int* ldCol;        ///< Leading dimensions of color slices.
    const int ldldCol;       ///< Leading dimension of ldCol.
    const int ldPtrCol;      ///< Leading dimension of pointer array.
    const double intensity;  ///< Intensity multiplier.

public:
    __device__ Crayon(int** colors, const int* ldCol, const int ldldCol, const int ldPtrCol, const Get& at, const double intensity): colors(colors), ldCol(ldCol), ldldCol(ldldCol), ldPtrCol(ldPtrCol), at(at), intensity(intensity){}

    /**
     * @brief Rounds the value and multiplies it by intensity.
     *
     * @param c The input color component.
     * @return The rounded intensity-scaled value.
     */
    __device__ int round(double c) const{
        return static_cast<int>(rint(c * intensity));
    }

    /**
     * @brief Stores an RGB color value, applying intensity scaling.
     * 
     * @param r Red component (0-255).
     * @param g Green component (0-255).
     * @param b Blue component (0-255).
     * @return Combined color.
     */
    __device__ int intColor(const double r, const double g, const double b) {
	return (round(r) << 16) | (round(g) << 8) | round(b);
    }
    /**
     * @brief Assigns the RGB color to the appropriate output location.
     * @param r Red component.
     * @param g Green component.
     * @param b Blue component.
     */
    __device__ void operator()(const int r, const int g, const int b){
    	at.set(colors, ldCol, ldldCol, ldPtrCol, intColor(r, g, b));
    }
};



/**
 * @brief CUDA kernel that assigns RGB colors to each pixel based on angles and optional intensity.
 * 
 * @param n Number of elements to process.
 * @param srcAngles Input angles in radians.
 * @param ldSrcAng Leading dimensions for srcAngles.
 * @param ldldSrc Stride in ldSrcAng.
 * @param ldPtrSrc Stride between srcAngles pointers.
 * @param heightSrcAng Height of the angle image.
 * 
 * @param colors Output RGB buffer.
 * @param ldCol Leading dimensions of the output buffer.
 * @param ldldCol Stride in ldCol columns.
 * @param ldPtrCol Stride between color pointer columns.
 * 
 * @param srcIntensities Optional input for intensity modulation.
 * @param ldSrcInt Leading dimensions for intensity.
 * @param ldldSrcInt Stride in ldSrcInt.
 * @param ldPtrSrcInt Stride for intensity pointers columns.
 * 
 * @param height Height of each slice.
 * @param width Width of each slice.
 * @param depth Number of slices per frame.
 */
extern "C" __global__ void colorsKernel(
    const int n, 
    
    const double** srcAngles, 
    const int* ldSrcAng, 
    const int ldldSrc,
    const int ldPtrSrc,
    const int heightSrcAng,
    
    int** colors,
    const int* ldCol,
    const int ldldCol,
    const int ldPtrCol,
    
    const double** srcIntensities,
    const int* ldSrcInt,
    const int ldldSrcInt,
    const int ldPtrSrcInt,
    
    const int height, const int width, const int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
 
    Get at(idx, height, width, depth); 
    
    double angle = at(srcAngles, ldSrcAng, ldldSrc, ldPtrSrc);
    
    double intensity = (ldldSrcInt == -1) ? 1.0 : at(srcIntensities, ldSrcInt, ldldSrcInt, ldPtrSrcInt);
    
    Crayon draw(colors, ldCol, ldldCol, ldPtrCol, at, intensity);
    
    if(isnan(angle)) draw(0, 0, 0);
    
    else if (-1e-5 <= angle && angle < CUDART_PI / 3) 
        draw(255, up(0, angle), 0);
    
    else if (CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI / 3) 
        draw(down(1, angle), 255, 0);
    
    else if (2 * CUDART_PI / 3 <= angle && angle < CUDART_PI) 
        draw(0, 255, up(2, angle));
    
    else if (CUDART_PI <= angle && angle < 4 * CUDART_PI / 3) 
        draw(0, down(3, angle), 255);
    
    else if (4 * CUDART_PI / 3 <= angle && angle < 5 * CUDART_PI / 3) 
        draw(up(4, angle), 0, 255);
    
    else if (5 * CUDART_PI / 3 <= angle && angle <= 2 * CUDART_PI + 1e-5) 
        draw(255, 0, down(5, angle));
    
    else draw(0, 0, 0);
    
}
