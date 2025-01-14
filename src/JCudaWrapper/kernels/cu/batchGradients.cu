/**
 * Represents multi-dimensional indices for a tensor element.
 */
class Indices {
public:
    const int* dim;       ///< Reference to tensor dimensions.
    const int gradient;   ///< Gradient index (0 = X, 1 = Y, 2 = Z).
    const int layer;      ///< Layer index in the tensor.
    const int col;        ///< Column index in the layer.
    const int row;        ///< Row index in the layer.
    const int flatIndex;  ///< Flat memory index.
    const double* data;   ///< Pointer to tensor data.

    /**
     * Constructs Indices from a flat index.
     * @param threadIndex Flat memory index.
     * @param dim Dimensions of the tensor batch.
     * @param data Pointer to tensor data.
     */
    __device__ Indices(int threadIndex, const int* dim, const double* data)
        : dim(dim),
          gradient(threadIndex / dim[6]),
          layer((threadIndex % dim[5]) / dim[4]),
          col((threadIndex % dim[4]) / dim[0]),
          row(threadIndex % dim[0]),
          flatIndex(threadIndex % dim[6]),
          data(data) {}

    /**
     * Computes a shifted index value in the gradient's direction.
     * @param offset Shift value.
     * @return Flat index of the shifted position.
     */
    __device__ double shift(int offset) const {
        int offsetIndex = 0;
        switch (gradient) {
            case 0: offsetIndex = flatIndex + dim[0] * offset; break;
            case 1: offsetIndex = flatIndex + offset; break;
            case 2: offsetIndex = flatIndex + dim[1] * dim[0] * offset; break;
        }
        return data[offsetIndex];
    }

    /**
     * Computes the gradient using a stencil method.
     * @return Computed gradient value.
     */
    __device__ double grad() const {
        int loc, end;
        switch (gradient) {
            case 0: loc = col; end = dim[1]; break;
            case 1: loc = row; end = dim[0]; break;
            case 2: loc = layer; end = dim[2]; break;
        }

        if (end == 1) return 0.0; // Single element case.

        if (loc == 0) return shift(1) - data[flatIndex]; // Forward difference at start.
        if (loc == end - 1) return data[flatIndex] - shift(-1); // Backward difference at end.
        if (loc == 1 || loc == end - 2) return (shift(1) - shift(-1)) / 2.0; // Central difference.
        return (shift(-2) - 8.0*shift(-1) + 8.0*shift(1) - shift(2))/12.0; // Higher-order stencil.
    }

    /**
     * Prints the current state of the indices for debugging.
     */
    __device__ void print() const {
        printf("FlatIndex: %d | Gradient: %d | Layer: %d | Col: %d | Row: %d | Dimensions [H: %d, W: %d, D: %d, N: %d]\n",
               flatIndex, gradient, layer, col, row,
               dim[0], dim[1], dim[2], dim[3]);
    }
};

/**
 * Kernel to compute gradients for batched tensors.
 * @param n Total number of elements in the batch.
 * @param mat Pointer to input tensor data.
 * @param dim indices height -> 0, width -> 1, depth -> 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
 * @param dX Output gradients along X direction.
 * @param dY Output gradients along Y direction.
 * @param dZ Output gradients along Z direction.
 */
extern "C" __global__ void batchGradientsKernel(
    const int n, 
    const double* mat, 
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
    double* dX, double* dY, double* dZ
) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx >= n) return;

    const Indices indices(idx, dim, mat);

    // Compute gradient and store in the corresponding output array.
    double gradValue = indices.grad();

    switch (indices.gradient) {
        case 0: dX[idx] = gradValue; break;
        case 1: dY[idx % dim[6]] = gradValue; break;
        case 2: dZ[idx % dim[6]] = gradValue; break;
    }
}

