
//please ensure that n = batchSize * height * width
extern "C" __global__ void prodScalarMatrixBatchKernel(int n, float *scalars, int inc, float *to, int height, int width, int colDist, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int matrixSize = height*width;
    
    if (idx < n) {
    
    	int fromIndex = idx / matrixSize;
    	int inMatrixIndex = idx % matrixSize;
    	
    	int colIndex = inMatrixIndex / height;
    	int rowIndex = inMatrixIndex % height;
        
        int toIndex = fromIndex * stride + colIndex * colDist + rowIndex;
        
        to[toIndex] *= scalars[fromIndex*inc]; 
    }
}
//nvcc -ptx fillMatrix.cu 
