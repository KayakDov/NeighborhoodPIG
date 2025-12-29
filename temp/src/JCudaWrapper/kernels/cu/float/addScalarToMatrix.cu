//The first value of from us used to fill this matrix.
//n is the size of the matrix, width * height.

extern "C" __global__ void addScalarToMatrixKernel(int n, float *from, int ldTo, float *to, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx >= n) return;
    
    to[(idx/height)*ldTo + idx % height] += from[0]; 
    
}
//nvcc -ptx addScalarToMatrix.cu 