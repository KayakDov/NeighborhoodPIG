#include <cmath>  // For std::round
#include <math_constants.h>

__device__ int down(int a, double theta){
    return static_cast<int>(std::round(255*(1 + a - 3*theta/CUDART_PI)));
}
__device__ int up(int a, double theta){
    return static_cast<int>(std::round(255*(3*theta/CUDART_PI - a)));
}


// ldColor must be greater than or equal to 3.
extern "C" __global__
void colorKernel(const double* angles, int anglesInc, int* colors, int ldColor, int n, double* intensities, int intensitiesInc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double angle = *(angles + idx * anglesInc);
    int* color = colors + idx * ldColor;
    double intensity = *(intensities + idx * intensitiesInc);

    if (0 <= angle && angle < CUDART_PI / 3) {
        color[0] = 255;
        color[1] = up(0, angle);
        color[2] = 0;    
    } else if (CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI / 3) {
        color[0] = down(1, angle);
        color[1] = 255; 
        color[2] = 0;   
    } else if (2 * CUDART_PI / 3 <= angle && angle < CUDART_PI) {
        color[0] = 0;  
        color[1] = 255;
        color[2] = up(2, angle);
    } else if (CUDART_PI <= angle && angle < 4 * CUDART_PI / 3) {
        color[0] = 0;
        color[1] = down(3, angle);
        color[2] = 255;
    } else if (4 * CUDART_PI / 3 <= angle && angle < 5 * CUDART_PI / 3) {
        color[0] = up(4, angle);
        color[1] = 0;
        color[2] = 255;
    } else if (5 * CUDART_PI / 3 <= angle && angle < 2 * CUDART_PI) {
        color[0] = 255;  
        color[1] = 0;    
        color[2] = down(5, angle);
    }

    color[0] *= intensity;//TODO: these should all be done at once.
    color[1] *= intensity;
    color[2] *= intensity;
}
