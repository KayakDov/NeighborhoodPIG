// CUDA Kernel to generate pointers in a GPU array
extern "C" __global__ void genPtrsKernel(int n, double *firstPointer, int shift, double** array, int inc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shift each pointer by the given right and down offsets
    if(idx < n){
        array[idx*inc] = shift*idx + firstPointer;
    }
}

