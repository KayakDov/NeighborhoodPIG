/**
 * Represents multi-dimensional indices for a tensor element.
 */
class Indices {
public:
    const int* dim;       ///< Reference to tensor dimensions.
    const int gradient;   ///< Gradient index (0 = X, 1 = Y, 2 = Z).
    const int srcFlatIndex;  ///< Flat memory index.
    const double* data;   ///< Pointer to tensor data.
    const int idx;

    /**
     * Constructs Indices from a flat index.
     * @param threadIndex Flat memory index.
     * @param dim Dimensions of the tensor batch.
     * @param data Pointer to tensor data.
     */
    __device__ Indices(int idx, const int* dim, const double* data)
        : dim(dim),
          gradient(idx / dim[6]),
          srcFlatIndex(((idx % dim[6]) / dim[0]) * dim[7] + idx % dim[0]),
          data(data),
          idx(idx) {}

    /**
     * Computes a shifted index value in the gradient's direction.
     * @param offset Shift value.
     * @return Flat index of the shifted position.
     */
    __device__ double shift(int offset) const {
        int offsetIndex;
        switch (gradient) {
            case 0: offsetIndex = srcFlatIndex + dim[7] * offset; break;
            case 1: offsetIndex = srcFlatIndex + offset; break;
            case 2: offsetIndex = srcFlatIndex + dim[1] * dim[7] * offset; break;
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
            case 0: loc = (idx % dim[4]) / dim[0]; end = dim[1]; break;
            case 1: loc = idx % dim[0];            end = dim[0]; break;
            case 2: loc = (idx % dim[5]) / dim[4]; end = dim[2]; break;
        }


//		if(idx == 161) printf("id = %d with gradient id %d, has layer %d \n", idx, gradient, (idx % dim[5]) / dim[4]);
//		if(idx >= 162) printf("id = %d with gradient id %d, has layer %d, note: tensor size = %d  and layer size = %d\n", idx, gradient, loc, dim[5], dim[4]);
	//	if(idx == 171) printf("id = %d with gradient id %d, has layer %d \n", idx, gradient, loc);

        if (end == 1) return 0.0; // Single element case.
        if (loc == 0) return shift(1) - data[srcFlatIndex]; // Forward difference at start.
        if (loc == end - 1) return data[srcFlatIndex] - shift(-1); // Backward difference at end.
        if (loc == 1 || loc == end - 2) return (shift(1) - shift(-1)) / 2.0; // Central difference.
        return (shift(-2) - 8.0*shift(-1) + 8.0*shift(1) - shift(2))/12.0; // Higher-order stencil.
    }

};

/**
 *Computes the indices written to.
 */
class DstIndices{
private:
	const int col;
	const int row;
public:
	/**
	 * @param height the height of the matrix.
	 * @param batchSize the number of elements in the batch.
	 * @param idx the thread id.
	 */
	__device__ DstIndices(int height, int batchSize, int idx): row(idx%height), col((idx % batchSize)/height){}
	/**
	 * Computes the index on the destination matrix.
	 * @param ldDst the leading dimension of the destination for which the index is computed.
	 * @param dInd the index in the batch (idx % batchSize).
	 * @param height the height of the matrix.
	 */
	__device__ int index(int ldDst) const {
		return col*ldDst + row;
	}
};
/**
 * Kernel to compute gradients for batched tensors.
 * @param n Total number of elements in the gradients.
 * @param mat Pointer to input tensor data.
 * @param dim indices height -> 0, width -> 1, depth -> 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
 * @param dX Output gradients along X direction.
 * @param dY Output gradients along Y direction.
 * @param dZ Output gradients along Z direction.
 */
extern "C" __global__ void batchGradientsKernel(
    const int n, 
    const double* mat, 
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6, ld = 7
    double* dX, const int ldx, double* dY, const int ldy, double* dZ, const int ldz
) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx >= n) return;

    const Indices indices(idx, dim, mat);

    const DstIndices to(dim[0], dim[6], idx);
     
    if(idx < dim[6])        dX[to.index(ldx)] = indices.grad();
    else if(idx < 2*dim[6]) dY[to.index(ldy)] = indices.grad();
    else                    dZ[to.index(ldz)] = indices.grad();
}

