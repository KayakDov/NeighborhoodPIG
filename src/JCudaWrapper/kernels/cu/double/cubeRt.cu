
extern "C" __global__ void cubeRtKernel(int n, double *from, int incFrom, double *to, int incTo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < n) {
        to[idx * incTo] = cbrt(from[idx * incFrom]); 
    }
}
//nvcc -ptx cubeRt.cu 
