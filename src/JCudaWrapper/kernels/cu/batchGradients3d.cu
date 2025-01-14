
//citation:  https://www.researchgate.net/publication/339542084_High-accuracy_compact_difference_schemes_for_differential_equations_in_mathematical_sciences
/**
 * Represents the dimensions and memory layout of batched tensors.
 */
class Dim {
public:
    const int &height, &width, &depth, &numTensors; ///< Tensor dimensions and batch size.
    const int layerSize;   ///< Total number of elements in a single layer.
    const int tensorSize;  ///< Total number of elements in a single tensor.
    const int batchSize;   ///< Total number of elements in the batch.

    /**
     * Constructor for Dim.
     * @param height Number of rows in each tensor layer.
     * @param width Number of columns in each tensor layer.
     * @param depth Number of layers per tensor.
     * @param numTensors Number of tensors in the batch.
     */
    __device__ Dim(int height, int width, int depth, int numTensors)
        : height(height), width(width), depth(depth), numTensors(numTensors),
          layerSize(height * width), tensorSize(layerSize * depth), batchSize(tensorSize * numTensors) {}

};

/**
 * Represents multi-dimensional indices for a tensor element.
 */
class Indices {
private:
    const Dim& dim;
    const int gradient; ///< Gradient index (0 = X, 1 = Y, 2 = Z).    
    const int layer;    ///< Layer index in the tensor.
    const int col;      ///< Column index in the layer.
    const int row;      ///< Row index in the layer.
    const int flatIndex; ///< Flat memory index.
    const double* data;
    
    __device__ Indices val(int dx, int dy, int dz) const {
        return data[flatIndex + dz*dim.layerSize + dx*dim.height + dy];
    }
    
    __device__ Indices val(int dir, int dist) const {
        switch(dir){
            case 0: return val(dist, 0, 0);
            case 1: return val(0, dist, 0);
            case 2: return val(0, 0, dist);
        }
    }
    
    /**
     * For dir, 0, 1, or 2 for x, y, or z. 
     */
    __device__ double sheetSum(int dir, int dist) const {
    
        int sum = 0;
        
        switch(dir){
            case 0: 
 		for(int dy = -1; dy <= 1; dy++)
 		    for(int dz = -1; dz <= 1; dz++)
 		        sum += val(dist, dy, dz);
 	        return sum;
            case 1: 
                for(int dx = -1; dx <= 1; dx++)
 		    for(int dz = -1; dz <= 1; dz++)
 		        sum += val(dx, dist, dz);
 	        return sum;
		
            case 2: 
		for(int dx = -1; dx <= 1; dx++)
 		    for(int dy = -1; dy <= 1; dy++)
 		        sum += val(dx, dy, dist);
 	        return sum;
        }
    }
public:
    /**
     * Constructs Indices from a flat index.
     * @param threadIndex Flat memory index.
     * @param dim Dimensions of the tensor batch.
     */
    __device__ Indices(int threadIndex, const Dim &dim, const double*data)
        : gradient(threadIndex / dim.batchSize),          
          layer((threadIndex % dim.tensorSize) / dim.layerSize),
          col((threadIndex % dim.layerSize) / dim.height),
          row(threadIndex % dim.height),
          flatIndex(threadIndex % dim.batchSize),
          dim(dim), data(data){}
    
    __device__ double grad(int dir){
        return 1.0/12*(-val(dir, 2) + 8*val(dir, 1) - 8*val(dir, -1) + val(dir, -2)) + 1.0/24*(sheetSum(dir, 1) - sheetSum(dir, -1));
    }
    
};


/**
 * Kernel to compute gradients for batched tensors.
 * @param n Total number of elements in the batch.
 * @param mat Pointer to input tensor data.
 * @param height Tensor height.
 * @param width Tensor width.
 * @param depth Tensor depth.
 * @param numTensors Number of tensors in the batch.
 * @param dX Output gradients along X direction.
 * @param dY Output gradients along Y direction.
 * @param dZ Output gradients along Z direction.
 */
extern "C" __global__ void batchGradientsKernel(
    const int n, 
    const double *mat, 
    const int height, const int width, const int depth, const int numTensors,
    double *dX, double *dY, double *dZ
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const Dim dim(height, width, depth, numTensors);
    const Indices indices(idx, dim);
    const Pixel pixel(indices, mat, dim);

    switch (indices.gradient) {
        case 0: dX[idx] = pixel.grad(); break;
        case 1: dY[idx % dim.batchSize] = pixel.grad(); break;
        case 2: dZ[idx % dim.batchSize] = pixel.grad(); break;
    }
}

