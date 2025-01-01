#include <cmath>  // For std::round
#include <math_constants.h>

/**
 * @brief Calculates the color intensity when transitioning downwards in the RGB spectrum.
 * @param a The base RGB component index (0-5) for the color spectrum.
 * @param theta The angle in radians.
 * @return The calculated intensity value (0-255) for the downward transition.
 */
__device__ int down(int a, double theta) {
    return static_cast<int>(std::round(255 * (1 + a - 3 * theta / CUDART_PI)));
}

/**
 * @brief Calculates the color intensity when transitioning upwards in the RGB spectrum.
 * @param a The base RGB component index (0-5) for the color spectrum.
 * @param theta The angle in radians.
 * @return The calculated intensity value (0-255) for the upward transition.
 */
__device__ int up(int a, double theta) {
    return static_cast<int>(std::round(255 * (3 * theta / CUDART_PI - a)));
}

/**
 * @class Writer
 * @brief A helper class to write RGB values to the output array.
 */
class Writer {
private:
    bool isTriplet;  ///< Flag to indicate if the output is a triplet (RGB components) or a packed integer.
    int* writeTo;    ///< Pointer to the output array.
    double intensity;///< Intensity scaling factor.

public:
    /**
     * @brief Constructor for the Writer class.
     * @param isTriplet Indicates whether the output is in triplet format.
     * @param writeTo Pointer to the output location in the array.
     * @param intensity Scaling factor for intensity.
     */
    __device__ Writer(bool isTriplet, int* writeTo, double intensity)
        : isTriplet(isTriplet), writeTo(writeTo), intensity(intensity) {}

    /**
     * @brief Computes a single packed RGB color value with optional intensity scaling.
     * @param r Red component (0-255).
     * @param g Green component (0-255).
     * @param b Blue component (0-255).
     */
    __device__ void setColor(int r, int g, int b) {
        int scaledR = std::round(r * intensity);
        int scaledG = std::round(g * intensity);
        int scaledB = std::round(b * intensity);

        if (isTriplet) {
            writeTo[0] = scaledR;
            writeTo[1] = scaledG;
            writeTo[2] = scaledB;
        } else {
            *writeTo = (scaledR << 16) | (scaledG << 8) | scaledB;
        }
    }
};

/**
 * @brief Kernel function to compute RGB color values based on input angles and intensities.
 * @details Each thread processes one angle, calculates an RGB color, and stores the result.
 * 
 * @param n The number of angles to process.
 * @param angles Pointer to the input array of angles (in radians).
 * @param anglesInc The increment between consecutive angles in the input array.
 * @param colors Pointer to the output array for storing packed RGB color values.
 * @param ldColors Leading dimension of the colors array. Must be 3 for triplets of colors, or 1 for packed RGB integers.
 * @param intensities Pointer to the input array of intensity values (optional).
 * @param intensitiesInc The increment between consecutive intensity values in the input array.
 *                      Pass -1 if intensities are not used.
 */
extern "C" __global__ void colorsKernel(
    int n, 
    const double* angles, 
    const int anglesInc, 
    int* colors,
    const int ldColors,
    const double* intensities, 
    int intensitiesInc
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Load the angle and intensity
    double angle = angles[idx * anglesInc];
    double intensity = (intensitiesInc != -1) ? intensities[idx * intensitiesInc] : 1.0;

    // Create a Writer object for the current thread
    Writer writer(ldColors == 3, colors + idx * ldColors, intensity);

    // Determine the RGB color based on the angle
    if (0 <= angle && angle < CUDART_PI / 3)
        writer.setColor(255, up(0, angle), 0);
    else if (CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI / 3)
        writer.setColor(down(1, angle), 255, 0);
    else if (2 * CUDART_PI / 3 <= angle && angle < CUDART_PI)
        writer.setColor(0, 255, up(2, angle));
    else if (CUDART_PI <= angle && angle < 4 * CUDART_PI / 3)
        writer.setColor(0, down(3, angle), 255);
    else if (4 * CUDART_PI / 3 <= angle && angle < 5 * CUDART_PI / 3)
        writer.setColor(up(4, angle), 0, 255);
    else if (5 * CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI)
        writer.setColor(255, 0, down(5, angle));
}

