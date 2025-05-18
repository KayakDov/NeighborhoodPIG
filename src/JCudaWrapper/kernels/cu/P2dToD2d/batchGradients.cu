


/**
 * Represents the multi-dimensional indices needed to access a specific element
 * in a batched tensor structure.
 */
class Indices {
public:
    const int idx;       ///< The global thread index.
    const int tensorInd; ///< The tensorindex (frame), unadjusted for leading dimension.
    const int layerInd; //< The layer index, unadjusted for leading dimension.
    const int* dim;     ///< Pointer to the tensor dimension array.

    /**
     * Computes the linear index (page offset) in the 2D matrix of pointers.
     * @param ldPtr Leading dimension of the layer matrix.
     * @return Offset into the pointer matrix.
     */
    __device__ int page(const int ldPtr) const {
    	return tensorInd * ldPtr + layerInd;
    }
    
    /**
     * Computes the offset into a single tensor layer.
     * @param ld Pointer to leading dimension array for each layer.
     * @param ldld Leading dimension of the ldMatrix itself.
     * @return Offset into the memory location for the element.
     */
    __device__ int word(const int* ld, const int ldld) const {
	return ((idx % dim[4]) / dim[0]) * ld[tensorInd * ldld + layerInd] + idx % dim[0];
    }

    /**
     * Constructs Indices from a flat thread index.
     * @param threadID Global thread index.
     * @param dim Array describing tensor shape:
     *        height → 0, width → 1, depth → 2, numTensors → 3,
     *        layerSize → 4, tensorSize → 5, batchSize → 6.
     */
    __device__ Indices(int threadID, const int* dim): 
	idx(threadID%dim[6]),
    	tensorInd(idx/dim[5]),
    	layerInd((idx/dim[4])%dim[2]),
    	dim(dim){}


};

/**
 * Utility to compute gradient of a tensor element using a finite difference stencil.
 */
class Grad{
private:
    const double** data; ///< Pointer to batched tensor data.
    const int page;      ///< Offset into the layer matrix.
    const int word;      ///< Offset within the tensor layer.
    
public:

    /**
     * Constructs a Grad object for a specific thread.
     * @param data Pointer to tensor data matrix.
     * @param inds Precomputed indices for the thread.
     * @param dim Tensor dimension array.
     * @param ld Pointer to leading dimension matrix.
     * @param ldld Leading dimension of ldMatrix.
     * @param ldPtr Leading dimension of the pointer matrix.
     */
    __device__ Grad(const double** data, const Indices& inds, const int* dim, const int* ld, const int ldld, const int ldPtr):
        data(data), 
        page(inds.page(ldPtr)), 
        word(inds.word(ld, ldld)){}

    /**
     * Computes a spatial gradient using a finite difference stencil.
     *
     * @param loc Index in the current dimension (x, y, or z).
     * @param end Size of the current dimension.
     * @param layerScale Scaling factor for spacing between layers.
     * @param dPage Offset for stepping through layers.
     * @param dWord Offset for stepping through positions within a layer.
     * @return Computed gradient value.
     */
    __device__ double at(const int loc, const int end, const double layerScale, const int dPage, const int dWord) const {

	double val;

        if (end == 1) val = 0.0; // Single element case.
        else if (loc == 0) val = data[page + dPage][word + dWord] - data[page][word]; // Forward difference at start.
        else if (loc == end - 1) val = data[page][word] - data[page - dPage][word - dWord]; // Backward difference at end.
        else if (loc == 1 || loc == end - 2) val = (data[page + dPage][word + dWord] - data[page - dPage][word - dWord]) / 2.0; // Central difference.
        else val = (data[page - 2*dPage][word - 2*dWord] - 8.0*data[page - dPage][word - dWord] + 8.0*data[page + dPage][word + dWord] - data[page + 2*dPage][word + 2*dWord])/12.0; // Higher-order stencil.
        
        return layerScale == 1? val : val/layerScale;
    }    

    

};
/**
 * Computes numerical gradients for a batch of 3D tensors using finite differences.
 * 
 * The input tensors are organized in a 2D array of pointers where each column is a tensor,
 * and each row corresponds to a layer. Each pointer in this matrix points to a 
 * height x width column-major matrix representing a layer. 
 * 
 * The gradients are computed along the X, Y, and Z axes and stored in output tensors.
 * 
 * @param n Total number of elements to process (should be 3 * height * width * depth * numTensors).
 * @param mat A depth x numTensors matrix of pointers to input tensor layers.
 * @param ldMat Leading dimensions for each tensor layer (column-major).
 * @param ldldMat Leading dimension of ldMat (needed to index correctly).
 * @param ldPtrMat Leading dimension of the pointer matrix `mat`.
 * @param dim Array of tensor dimensions:
 *            - dim[0] = height
 *            - dim[1] = width
 *            - dim[2] = depth
 *            - dim[3] = numTensors
 *            - dim[4] = layerSize (height * width)
 *            - dim[5] = tensorSize (depth * height * width)
 *            - dim[6] = batchSize (3 * tensorSize)
 * @param dX Gradient outputs in the X direction.
 * @param ldx Leading dimensions for dX.
 * @param ldldX Leading dimension of ldx.
 * @param ldPtrX Leading dimension of pointer matrix dX.
 * @param dY Gradient outputs in the Y direction.
 * @param ldy Leading dimensions for dY.
 * @param ldldY Leading dimension of ldy.
 * @param ldPtrY Leading dimension of pointer matrix dY.
 * @param dZ Gradient outputs in the Z direction.
 * @param ldz Leading dimensions for dZ.
 * @param ldldZ Leading dimension of ldz.
 * @param ldPtrZ Leading dimension of pointer matrix dZ.
 * @param zLayerMult Scaling factor for the z-gradient (accounts for spacing differences between z-layers and x/y pixels).
 */
extern "C" __global__ void batchGradientsKernel(
    const int n, 
    const double** mat, const int* ldMat, const int ldldMat, const int ldPtrMat,
    double** dX, const int* ldx, const int ldldX, const int ldPtrX,
    double** dY, const int* ldy, const int ldldY, const int ldPtrY,
    double** dZ, const int* ldz, const int ldldZ, const int ldPtrZ,
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
    const double zLayerMult
) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx >= n) return;

    printf("Ahoy from batchGradients!");

    const Indices inds(idx, dim);
    const Grad grad(mat, inds, dim, ldMat, ldldMat, ldPtrMat);

    switch(idx / dim[6]){ 
    	case 0: dX[inds.page(ldPtrX)][inds.word(ldx, ldldX)] 
    		= grad.at((idx % dim[4]) / dim[0], dim[1],  1,          0, ldMat[inds.page(ldPtrMat)]); break;
	case 1: dY[inds.page(ldPtrY)][inds.word(ldy, ldldY)] 
		= grad.at(idx % dim[0],            dim[0],  1,          0, 1                      ); break;
	case 2: dZ[inds.page(ldPtrZ)][inds.word(ldz, ldldZ)] 
		= grad.at((idx % dim[5]) / dim[4], dim[2],  zLayerMult, 1, 0                      );
    }
}

