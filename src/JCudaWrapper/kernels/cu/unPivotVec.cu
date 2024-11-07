//#include <cstdio>  // for printf support in CUDA

__device__ void swap(double* vec, int i, int j){
    if (i == j) return;

    double temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

//We assume pivoted describes square matrices: width = height.
//n should be the total number of columns;
extern "C" __global__ void unPivotVecKernel(int *pivotScheme, int dim, double *pivoted, int inc, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;

    int vecsStartInd = stride*idx;
    int psStartIndex = dim*(idx/dim);

    for (int i = dim - 1; i >= 0; i--) swap(pivoted, vecsStartInd + i*inc, vecsStartInd + (pivotScheme[psStartIndex + i] - 1)*inc);
}


//nvcc -ptx unPivot.cu 



//printf("Thread %d.\n",idx);
//printf("Thread %d:pivoting index %d and index %d \n", idx, currentPos, pivotIdx);
