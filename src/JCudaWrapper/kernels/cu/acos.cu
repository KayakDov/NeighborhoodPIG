
extern "C" __global__ void acosKernel(int n, double *from, int incFrom, double *to, int incTo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx >= n) return;
        
    to[idx * incTo] = acos(from[idx * incFrom]); 
    
}
//nvcc -ptx acos.cu 
