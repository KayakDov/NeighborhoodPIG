#include <cmath>  // For std::round
#include <math_constants.h>

__device__ int down(int a, double theta) {
    return static_cast<int>(std::round(255 * (1 + a - 3 * theta / CUDART_PI)));
}
__device__ int up(int a, double theta) {
    return static_cast<int>(std::round(255 * (3 * theta / CUDART_PI - a)));
}

// ldColor is no longer used since we store the result in a single integer.
// Pass -1 for intensitiesInc if intensities are not to be used.
extern "C" __global__
void colorSingleKernel(int n, const double* angles, int anglesInc, int* colors, double* intensities, int intensitiesInc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double angle = *(angles + idx * anglesInc);
    int r = 0, g = 0, b = 0;  // Initialize RGB values.

    if (0 <= angle && angle < CUDART_PI / 3) {
        r = 255;
        g = up(0, angle);
        b = 0;    
    } else if (CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI / 3) {
        r = down(1, angle);
        g = 255; 
        b = 0;   
    } else if (2 * CUDART_PI / 3 <= angle && angle < CUDART_PI) {
        r = 0;  
        g = 255;
        b = up(2, angle);
    } else if (CUDART_PI <= angle && angle < 4 * CUDART_PI / 3) {
        r = 0;
        g = down(3, angle);
        b = 255;
    } else if (4 * CUDART_PI / 3 <= angle && angle < 5 * CUDART_PI / 3) {
        r = up(4, angle);
        g = 0;
        b = 255;
    } else if (5 * CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI) {
        r = 255;  
        g = 0;    
        b = down(5, angle);
    }

    if (intensitiesInc != -1) {
        double intensity = *(intensities + idx * intensitiesInc);
        r = static_cast<int>(r * intensity);
        g = static_cast<int>(g * intensity);
        b = static_cast<int>(b * intensity);
    }

    // Pack RGB into a single integer (0xRRGGBB format).
    colors[idx] = (r << 16) | (g << 8) | b;
}
