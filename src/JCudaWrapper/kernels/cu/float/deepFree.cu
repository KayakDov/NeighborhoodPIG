
extern "C" __global__ void deepFreeKernel(int numPointers, void** devicePointers) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPointers)  return;
    
    cudaFree(devicePointers[idx]);
    
}