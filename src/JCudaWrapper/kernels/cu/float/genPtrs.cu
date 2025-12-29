// CUDA Kernel to generate pointers in a GPU array
extern "C" __global__ void genPtrsKernel(int n, float *firstPointer, int shift, float** array, int inc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < n){
        array[idx*inc] = shift*idx + firstPointer;
    }
}

