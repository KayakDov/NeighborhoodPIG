/**
 * CUDA kernel that fills specified positions in an array with a given value.
 *
 *
 * @param n       The total number of elements to process.
 * @param array   Pointer to the array to be modified.
 * @param inc     Stride increment for indexing.
 * @param height  The height dimension used in index transformation. 
 * If this method is being called on a 1d array, than this can just be the 
 * length of the array.
 * @param ldFrom  Leading dimension for the transformation.  
 * Should be greater than or equal to the height and may be used to compensate 
 * for pitch.
 * @param fill    The value to be assigned to the computed indices.
 */
extern "C" __global__ void fillKernel(int n, double *array, int inc, int height, double ldFrom, double fill) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < n) {
        idx *= inc;
        idx = (idx / height) * ldFrom + idx % height;
        array[idx] = fill; 
    }
}
