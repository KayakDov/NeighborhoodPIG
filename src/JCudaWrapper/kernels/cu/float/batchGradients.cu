/**
 * Represents multi-dimensional indices for a tensor element.
 */
class Indices {
public:
    const float* data;   ///< Pointer to tensor data.

    /**
     * Constructs Indices from a flat index.
     * @param threadIndex Flat memory index.
     * @param dim Dimensions of the tensor batch.
     * @param data Pointer to tensor data.
     */
    __device__ Indices(int idx, const int* dim, const float* data)
        : data(data + ((idx % dim[6]) / dim[0]) * dim[7] + idx % dim[0]) {}

    /**
     * Computes the gradient using a stencil method.
     * @param layerScale How many times greater is the physical distance between z layers than the real world distance between pixels in the xy plane.
     * @param pixInc The increment between adjacent pixels in the current dimension.
     * @return Computed gradient value.
     */
    __device__ float grad(int loc, int end, float layerScale, int pixInc) const {

		float val;

        if (end == 1) val = 0.0; // Single element case.
        else if (loc == 0) val = data[pixInc] - data[0]; // Forward difference at start.
        else if (loc == end - 1) val = data[0] - data[-pixInc]; // Backward difference at end.
        else if (loc == 1 || loc == end - 2) val = (data[pixInc] - data[-pixInc]) / 2.0; // Central difference.
        else val = (data[-2*pixInc] - 8.0*data[-pixInc] + 8.0*data[pixInc] - data[2*pixInc])/12.0; // Higher-order stencil.
        
        return layerScale == 1? val : val/layerScale;
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
    const float* mat, 
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6, ld = 7
    float* dX, const int ldx, float* dY, const int ldy, float* dZ, const int ldz,
    const float zLayerMult
) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx >= n) return;

    const Indices indices(idx, dim, mat);

    const DstIndices to(dim[0], dim[6], idx);
    switch(idx / dim[6]){ 
    	case 0: dX[to.index(ldx)] = indices.grad((idx % dim[4]) / dim[0], dim[1], 1, dim[7]); break;
		case 1: dY[to.index(ldy)] = indices.grad(idx % dim[0],  dim[0], 1, 1); break;
		case 2: dZ[to.index(ldz)] = indices.grad((idx % dim[5]) / dim[4], dim[2], zLayerMult, dim[1] * dim[7]);
    }
}

