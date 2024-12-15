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
public:
    const Dim& dim;
    const int gradient; ///< Gradient index (0 = X, 1 = Y, 2 = Z).    
    const int layer;    ///< Layer index in the tensor.
    const int col;      ///< Column index in the layer.
    const int row;      ///< Row index in the layer.
    const int flatIndex; ///< Flat memory index.
    

    /**
     * Constructs Indices from a flat index.
     * @param threadIndex Flat memory index.
     * @param dim Dimensions of the tensor batch.
     */
    __device__ Indices(int threadIndex, const Dim &dim)
        : gradient(threadIndex / dim.batchSize),          
          layer((threadIndex % dim.tensorSize) / dim.layerSize),
          col((threadIndex % dim.layerSize) / dim.height),
          row(threadIndex % dim.height),
          flatIndex(threadIndex % dim.batchSize),
          dim(dim){}

    /**
     * Shifts the current indices along a specified dimension.
     * @param offset Shift value.
     * @param dimension Dimension to shift (0 = row, 1 = col, 2 = layer).
     * @return Flat index of the shifted position.
     */
    __device__ int shift(int offset) const {
        switch (gradient) {
            case 0: return flatIndex + dim.height*offset;
            case 1: return flatIndex + offset;
            case 2: return flatIndex + dim.width*dim.height*offset;
        }
        printf("bad gradient = %d", gradient);
    }
    
	/**
 	* Prints the current state of the indices in a single printf statement.
 	*/
	__device__ void print() const {
	    printf("FlatIndex: %d | Gradient: %d (0=X, 1=Y, 2=Z) | Layer: %d | Col: %d | Row: %d | Dimensions [H: %d, W: %d, D: %d, N: %d]\n",
	           flatIndex, gradient, layer, col, row,
	           dim.height, dim.width, dim.depth, dim.numTensors);
	}

};

/**
 * Encapsulates operations on a single tensor element.
 */
class Pixel {
public:
    const Indices &indices; ///< Indices of the current pixel.
    const Dim &dim;         ///< Dimensions of the tensor batch.
    const double *data;     ///< Pointer to tensor data.

    /**
     * Constructor for Pixel.
     * @param indices Indices of the pixel.
     * @param data Pointer to tensor data.
     * @param dim Dimensions of the tensor batch.
     */
    __device__ Pixel(const Indices &indices, const double *data, const Dim &dim)
        : indices(indices), data(data), dim(dim) {}

    /**
     * Shifts the current pixel along a specified dimension.
     * @param offset Shift value.
     * @param dimension Dimension to shift (0 = row, 1 = col, 2 = layer).
     * @return Value of the shifted pixel.
     */
    __device__ double shift(int offset) const {
        return data[indices.shift(offset)];
    }

    /**
     * Computes the gradient along a specified direction.     
     * @return Gradient value.
     */
    __device__ double grad() const {
        int loc, end;
        switch (indices.gradient) {
            case 0: loc = indices.col; end = dim.width; break;
            case 1: loc = indices.row; end = dim.height; break;
            case 2: loc = indices.layer; end = dim.depth; break;
        }
        
        //printf("The gradient is %d, the index is %d. The value is: %f, and shifted 0 is %f, and shifted -1 is %f, and shifted +1 is %f.\n", indices.gradient, indices.flatIndex, data[indices.flatIndex], shift(0), shift(-1), shift(1));

	//indices.print();

        if (end == 1) return 0;

        if (loc == 0) return shift(1) - data[indices.flatIndex];
        else if (loc == 1 || loc == end - 2) return (shift(1) - shift(-1)) / 2.0;
        else if (loc < end - 2) return (-shift(-2) / 12 + shift(-1) * 2.0 / 3 - shift(1) * 2.0 / 3 + shift(2) / 12);
        else return data[indices.flatIndex] - shift(-1);
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
    int n, const double *mat, int height, int width, int depth, int numTensors,
    double *dX, double *dY, double *dZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const Dim dim(height, width, depth, numTensors);//TODO: Is there a way to do this with shared memory?
    const Indices indices(idx, dim);
    const Pixel pixel(indices, mat, dim);

    switch (indices.gradient) {
        case 0: dX[idx] = pixel.grad(); break;
        case 1: dY[idx % dim.batchSize] = pixel.grad(); break;
        case 2: dZ[idx % dim.batchSize] = pixel.grad(); break;
    }
}

