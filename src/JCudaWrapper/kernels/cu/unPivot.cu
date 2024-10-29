

__device__ void swap(double* vec, int i, int j){
    if(i == j) return;
    double temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

extern "C" __global__ void unPivotKernel(int *pivotScheme, int numRows, double *pivoted, int ldP, int n) {
    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;  

    for(int i = numRows * idx - 1 ; i >= (numRows - 1) * idx; i--)
        swap(pivoted, idx*ldP + i, idx*ldP + pivotScheme[i]);
}

//nvcc -ptx unPivot.cu 