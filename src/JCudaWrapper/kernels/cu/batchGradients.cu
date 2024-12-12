class Indices;

/**
 * Represents the dimensions and memory layout of batched tensors.
 */
class Dim {
public:
    const int &height, &width, &depth, &numTensors;
    const int layerSize, tensorSize, batchSize;

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

    /**
     * Computes the flattened index in the tensor batch.
     * @param indices The multi-dimensional indices.
     * @return Flattened memory index.
     */
    __device__ int index(const Indices &indices) const;
};

/**
 * Represents multi-dimensional indices for a tensor element.
 */
class Indices {
public:
    const int gradient, tensor, layer, col, row;

    /**
     * Constructs Indices from a flat index.
     * @param flatIndex Flat memory index.
     * @param dim Dimensions of the tensor batch.
     */
    __device__ Indices(int flatIndex, const Dim &dim)
        : gradient(flatIndex / dim.batchSize),
          tensor((flatIndex % dim.batchSize) / dim.tensorSize),
          layer((flatIndex % dim.tensorSize) / dim.layerSize),
          col((flatIndex % dim.layerSize) / dim.height),
          row(flatIndex % dim.height) {}

    /**
     * Constructs Indices directly from gradient, tensor, layer, column, and row.
     */
    __device__ Indices(int gradient, int tensor, int layer, int col, int row)
        : gradient(gradient), tensor(tensor), layer(layer), col(col), row(row) {}

    /**
     * Shifts the current indices along a specified dimension.
     * @param offset Shift value.
     * @param dimension Dimension to shift (0 = row, 1 = col, 2 = layer).
     * @return New shifted Indices.
     */
    __device__ Indices shift(int offset, int dimension) const {
        switch (dimension) {
            case 0: return Indices(gradient, tensor, layer, col, row + offset);
            case 1: return Indices(gradient, tensor, layer, col + offset, row);
            case 2: return Indices(gradient, tensor, layer + offset, col, row);
            default: return *this; // No shift for invalid dimension
        }
    }
};

__device__ int Dim::index(const Indices &indices) const {
        return batchSize * indices.gradient +
               tensorSize * indices.tensor +
               layerSize * indices.layer +
               indices.col * height + indices.row;
    }

/**
 * Encapsulates operations on a single tensor element.
 */
class Pixel {
public:
    const Indices &indices;
    const Dim &dim;
    const double *data;

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
    __device__ double shift(int offset, int dimension) const {
        return data[dim.index(indices.shift(offset, dimension))];
    }

    /**
     * Computes the gradient along a specified direction.
     * @param dir Direction for gradient computation (0 = x, 1 = y, 2 = z).
     * @return Gradient value.
     */
    __device__ double grad(int dir) const {
        int loc, end;
        switch (dir) {
            case 0: loc = indices.col; end = dim.width; break;
            case 1: loc = indices.row; end = dim.height; break;
            case 2: loc = indices.layer; end = dim.depth; break;
            default: return 0.0; // Invalid direction
        }

        if (loc == 0)
            return shift(1, dir) - data[dim.index(indices)];
        else if (loc == 1 || loc == end - 2)
            return (shift(1, dir) - shift(-1, dir)) / 2.0;
        else if (loc < end - 2)
            return (-shift(-2, dir) / 12 + shift(-1, dir) * 2.0 / 3 -
                    shift(1, dir) * 2.0 / 3 + shift(2, dir) / 12);
        else
            return data[dim.index(indices)] - shift(-1, dir);
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

    Dim dim(height, width, depth, numTensors);
    Indices indices(idx, dim);
    Pixel pixel(indices, mat, dim);

    switch (indices.gradient) {
        case 0: dX[idx] = pixel.grad(0); break;
        case 1: dY[idx % dim.batchSize] = pixel.grad(1); break;
        case 2: dZ[idx % dim.batchSize] = pixel.grad(2); break;
    }
}

