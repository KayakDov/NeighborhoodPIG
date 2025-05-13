/**
 * Represents the multi-dimensional indices needed to access a specific element
 * in a batched tensor structure.
 */
class Indices {
public:  
    const int tensorInd; ///< The tensorindex (frame), unadjusted for leading dimension.
    const int layerInd; //< The layer index, unadjusted for leading dimension.
    
    /**
     * Computes the linear index (page offset) in the 2D matrix of pointers.
     * @param ldPtr Leading dimension of the layer matrix.
     * @return Offset into the pointer matrix.
     */
    __device__ int page(const int ldPtr) const {
    	return tensorInd * ldPtr + layerInd;
    }
  
    /**
     * Constructs Indices from a flat thread index.
     * @param threadID Global thread index.
     * @param dim Array describing tensor shape:
     *        height → 0, width → 1, depth → 2, numTensors → 3,
     *        layerSize → 4, tensorSize → 5, batchSize → 6.
     */
    __device__ Indices(int idx, const int* dim): 	
    	tensorInd(idx/dim[5]),
    	layerInd((idx/dim[4])%dim[2]){}


};



extern "C" __global__ void deepFreeKernel(
    const int numPointers, 
    void** mat, const int* ldMat, const int ldldMat, const int ldPtrMat,
    const int* dim
    ) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPointers)  return;
    
    Indices at(idx, dim);
    
    cudaFree(mat[at.page(ldPtrMat)]);
    
}
