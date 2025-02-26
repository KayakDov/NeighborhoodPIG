#include <cmath>  // For std::round
#include <math_constants.h>

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
 * @class Writer
 * @brief A helper class that writes RGB values to the output array.
 */
class Writer {
private:    
    int* writeTo;     ///< Pointer to the output location in the colors array.
    double intensity; ///< Intensity scaling factor.

public:
    /**
     * @brief Constructs a Writer object to store RGB values.
     * 
     * @param writeTo Pointer to the output location in the colors array.
     * @param intensity Scaling factor for intensity (1.0 if no scaling is applied).
     */
    __device__ Writer(int* writeTo, double intensity)
        : writeTo(writeTo), intensity(intensity) {}

    /**
     * @brief Rounds the value and multiplies it by intensity.
     *
     * @param c The input color component.
     * @return The rounded intensity-scaled value.
     */
    __device__ int roundValue(double c) {
        return static_cast<int>(rint(c * intensity));
    }

    /**
     * @brief Stores an RGB color value, applying intensity scaling.
     * 
     * @param r Red component (0-255).
     * @param g Green component (0-255).
     * @param b Blue component (0-255).
     */
    __device__ void setColor(double r, double g, double b) {
        *writeTo = (roundValue(r) << 16) | (roundValue(g) << 8) | roundValue(b);
    }
};

/**
 * @brief CUDA kernel to compute RGB color values from input angles and intensities.
 * 
 * @details This kernel processes n input angles, converting each to an RGB color based on its position in the spectrum.
 *          The results are stored in column-major order. Optionally, intensity scaling is applied.
 * 
 * @param n The number of input angles to process.
 * @param srcAngles Pointer to the column-major input array of angles (in radians).
 * @param ldSrcAng Leading dimension (stride) of srcAngles in memory.
 * @param heightSrcAng Number of rows in srcAngles (defines how many values belong to one column).
 * @param colors Pointer to the column-major output array for storing RGB values.
 * @param ldCol Leading dimension (stride) of colors in memory.
 * @param heightCol Number of rows in colors (defines how many values belong to one column).
 * @param srcIntensities Pointer to the column-major array of intensity values (optional).
 * @param heightSrcInt Number of rows in srcIntensities. Ignored if srcIntensities == nullptr.
 * @param ldSrcInt Leading dimension (stride) of srcIntensities. Pass -1 if srcIntensities is unused.
 */
extern "C" __global__ void colorsKernel(
    const int n, 
    
    const double* srcAngles, 
    const int ldSrcAng, 
    const int heightSrcAng,
    
    int* colors,
    const int ldCol,
    const int heightCol,
    
    const double* srcIntensities,
    const int heightSrcInt,
    const int ldSrcInt    
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    double angle = srcAngles[ind(idx, ldSrcAng, heightSrcAng)];    
    
    double intensity = (ldSrcInt == -1) ? 1.0 : srcIntensities[ind(idx, ldSrcInt, heightSrcInt)];
    
    Writer writer(colors + ind(idx, ldCol, heightCol), intensity);
    
    
    if(isnan(angle)) writer.setColor(0, 0, 0);
	else if (-1e-5 <= angle && angle < CUDART_PI / 3) 
        writer.setColor(255, up(0, angle), 0);
    
    else if (CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI / 3) 
        writer.setColor(down(1, angle), 255, 0);
    
    else if (2 * CUDART_PI / 3 <= angle && angle < CUDART_PI) 
        writer.setColor(0, 255, up(2, angle));
    
    else if (CUDART_PI <= angle && angle < 4 * CUDART_PI / 3) 
        writer.setColor(0, down(3, angle), 255);
    
    else if (4 * CUDART_PI / 3 <= angle && angle < 5 * CUDART_PI / 3) 
        writer.setColor(up(4, angle), 0, 255);
    
    else if (5 * CUDART_PI / 3 <= angle && angle <= 2 * CUDART_PI + 1e-5) 
        writer.setColor(255, 0, down(5, angle));
    else writer.setColor(0, 0, 0);
}
