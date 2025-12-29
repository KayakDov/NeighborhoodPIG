//#include <cstdio>  // for printf support in CUDA

__device__ void swap(float* vec, int i, int j){
    if (i == j) return;

    float temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

//We assume pivoted describes square matrices: width = height.
//n should be the total number of columns; n = batchSize * width.
extern "C" __global__ void unPivotMatrixKernel(int n, int *pivotScheme, int height, float *pivoted, int ldP) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;

    int startCol = ldP*idx;
    int startPivotSchemeRead = (idx/height)*height;
  

    for (int i = height - 1; i >= 0; i--) swap(pivoted, startCol + i, startCol + pivotScheme[startPivotSchemeRead + i] - 1);

    
}


//nvcc -ptx unPivot.cu 



//printf("Thread %d.\n",idx);
//printf("Thread %d:pivoting index %d and index %d \n", idx, currentPos, pivotIdx);
