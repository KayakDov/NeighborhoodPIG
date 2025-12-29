
extern "C" __global__ void sqrtKernel(int n, float *from, int incFrom, float *to, int incTo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < n) {
        to[idx * incTo] = sqrt(from[idx * incFrom]); 
    }
}
//nvcc -ptx sqrt.cu 