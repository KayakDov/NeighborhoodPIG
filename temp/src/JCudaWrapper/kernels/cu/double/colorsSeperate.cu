#include <cmath>          // For rounding
#include <math_constants.h> // For CUDART_PI

/**
 * @brief Calculates the color intensity when transitioning downwards in the RGB spectrum.
 * @param a The base RGB component index (0-5) for the color spectrum.
 * @param theta The angle in radians.
 * @return The calculated intensity value (0-255) for the downward transition.
 */
__device__ int down(int a, double theta) {
    return static_cast<int>(__double2int_rn(255 * (1 + a - 3 * theta / CUDART_PI)));
}

/**
 * @brief Calculates the color intensity when transitioning upwards in the RGB spectrum.
 * @param a The base RGB component index (0-5) for the color spectrum.
 * @param theta The angle in radians.
 * @return The calculated intensity value (0-255) for the upward transition.
 */
__device__ int up(int a, double theta) {
    return static_cast<int>(__double2int_rn(255 * (3 * theta / CUDART_PI - a)));
}

/**
 * @class Writer
 * @brief A helper class to write RGB values to the output arrays.
 */
class Writer {
private:
    unsigned char* red;    ///< Pointer to the red component array.
    unsigned char* green;  ///< Pointer to the green component array.
    unsigned char* blue;   ///< Pointer to the blue component array.
    double intensity; ///< Intensity scaling factor.

public:
    /**
     * @brief Constructor for the Writer class.
     * @param reds Pointer to the red component output array.
     * @param greens Pointer to the green component output array.
     * @param blues Pointer to the blue component output array.
     * @param intensity Scaling factor for intensity.
     */
    __device__ Writer(unsigned char* red, unsigned char* green, unsigned char* blue, double intensity)
        : red(red), green(green), blue(blue), intensity(intensity) {}

    /**
     * @brief Sets the RGB color values with intensity scaling.
     * @param r Red component (0-255).
     * @param g Green component (0-255).
     * @param b Blue component (0-255).
     */
    __device__ void setColor(int r, int g, int b) {
        *red = static_cast<unsigned char>(r * intensity);
        *green = static_cast<unsigned char>(g * intensity);
        *blue = static_cast<unsigned char>(b * intensity);
    }
};


/**
 * @brief Kernel function to compute RGB color values based on input angles and intensities.
 * 
 * @param n The number of angles to process.
 * @param angles Pointer to the input array of angles (in radians).
 * @param anglesInc The increment between consecutive angles in the input array.
 * @param reds Pointer to the red component output array.
 * @param greens Pointer to the green component output array.
 * @param blues Pointer to the blue component output array.
 * @param intensities Pointer to the input array of intensity values (optional).
 * @param intensitiesInc The increment between consecutive intensity values in the input array.
 *                      Pass -1 if intensities are not used.
 */
extern "C" __global__ void colorsSeperateKernel(
    int n, 
    const double* angles, const int anglesInc, 
    unsigned char* reds, unsigned char* greens, unsigned char* blues,
    const double* intensities, int intensitiesInc
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Load the angle and intensity
    double angle = angles[idx * anglesInc];
    double intensity = (intensitiesInc != -1) ? intensities[idx * intensitiesInc] : 1.0;

    // Create a Writer object for the current thread
    Writer writer(reds + idx, greens + idx, blues + idx, intensity);

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

