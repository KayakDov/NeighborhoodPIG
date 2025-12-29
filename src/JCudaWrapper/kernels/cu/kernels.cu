#include <cuda_runtime.h>
#include <math.h>

/**
 * Represents the multi-dimensional indices needed to access a specific element
 * in a batched tensor structure.
 */
class Indices {
public:
    const int idx;       ///< The global flat index.
    const int tensorInd; ///< The tensorindex (frame), unadjusted for leading dimension.
    const int layerInd;  //< The layer index, unadjusted for leading dimension.
    const int row;
    const int col;

    /**
     * Computes the flat index in the 2D matrix of pointers.
     * @param ldPtr Leading dimension of the layer matrix.
     * @return Offset into the pointer matrix.
     */
    __device__ int page(const int ldPtr) const {
    	return tensorInd * ldPtr + layerInd;
    }
    
    /**
     * Computes the flat index in the layer.
     * @param ld Pointer to leading dimension array for each layer.
     * @param ldld Leading dimension of the ldMatrix itself.
     * @return Offset into the memory location for the element.
     */
    __device__ int word(const int* ld, const int ldld) const {
        return col * ld[page(ldld)] + row;
    }

    /**
     * @brief Retrieves a value from the source 4D dataset using the calculated indices.
     *
     * @param src   Array of pointers to 2D slices, arranged in frame-major and then depth-major order.
     * @param ld    Array of leading dimensions for each slice (used for column-major indexing).
     * @param ldld  Leading dimension of the ld array (stride across layers).
     * @param ldPtr Leading dimension of the src array (stride across frames).
     * @return      The double value at the resolved position in the 4D dataset.
     */
    template<typename T>
    __device__ T operator()(const T** src, const int* ld, const int ldld, const int ldPtr) const{
        return src[page(ldPtr)][word(ld, ldld)];
    }
    template<typename T>
    __device__ T& operator()(T** src, const int* ld, const int ldld, const int ldPtr) {
        return src[page(ldPtr)][word(ld, ldld)];
    }

    /**
     * Constructs Indices from a flat thread index.
     * @param idx Global thread index.
     * @param dim Array describing tensor shape:
     *        height → 0, width → 1, depth → 2, numTensors → 3,
     *        layerSize → 4, tensorSize → 5, batchSize → 6.
     */
    __device__ Indices(int threadID, const int* dim): 
	idx(threadID % dim[6]),
    	tensorInd((idx % dim[6]) / dim[5]),
    	layerInd((idx % dim[5]) / dim[4]),
    	row(idx % dim[0]),
    	col((idx % dim[4])/ dim[0]){}

     __device__ void print() const {
     
        int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;

        printf("Thread %d (Global): Indices - idx: %d, tensorInd: %d, layerInd: %d, row: %d, col: %d\n",
               globalThreadId, idx, tensorInd, layerInd, row, col);
    }

    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     * @param downSampleFactorXY The downsampling factor applied in the x and y dimensions.
     * @param downSampleFactorZ The downsampling factor applied in the z dimension.
     */
    __device__ Indices(const int idx, const int* dim, const int downSampleFactorXY, const int downSampleFactorZ)
    : idx(idx),
      layerInd(((idx / dim[4]) % dim[2]) * downSampleFactorZ),
      tensorInd(idx / dim[5]),
      row((idx % dim[0])*downSampleFactorXY),
      col(((idx % dim[4])/dim[0])*downSampleFactorXY) {}


    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     * @param downSampleFactorXY The downsampling factor applied in the x and y dimensions.
     */
    __device__ Indices(const int inputIdx, const int* dim, const int downSampleFactorXY)
    : idx(inputIdx),
      tensorInd(idx / dim[4]),
      layerInd(0),
      row((idx % dim[0]) * downSampleFactorXY),
      col(((idx % dim[4])/dim[0]) * downSampleFactorXY) {}
};

/**
 * Utility to compute gradient of a tensor element using a finite difference stencil.
 */
class Grad{
private:
    const float** data; ///< Pointer to batched tensor data.
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
    __device__ Grad(const float** data, const Indices& inds, const int* dim, const int* ld, const int ldld, const int ldPtr):
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

        if (end == 1)                        val = 0.0; // Single element case.
        else if (loc == 0)                   val = data[page + dPage][word + dWord] - data[page][word]; // Forward difference at start.
        else if (loc == end - 1)             val = data[page][word] - data[page - dPage][word - dWord]; // Backward difference at end.
        else if (loc == 1 || loc == end - 2) val = (data[page + dPage][word + dWord] - data[page - dPage][word - dWord]) / 2.0; // Central difference.
        else                                 val = (data[page - 2*dPage][word - 2*dWord] - 8.0*data[page - dPage][word - dWord] + 8.0*data[page + dPage][word + dWord] - data[page + 2*dPage][word + 2*dWord])/12.0; // Higher-order stencil.
   
        return layerScale == 1? val : val/layerScale;
    }    

    __host__ __device__ void print() const {
        printf("Thread [%d]: Grad - page: %d, word: %d\n",
            threadIdx.x + blockIdx.x * blockDim.x,
            page, word);
    }



};

__device__ void batchGradients(const int n, 
    const float** mat, const int* ldMat, const int ldldMat, const int ldPtrMat,
    float** dX, const int* ldx, const int ldldX, const int ldPtrX,
    float** dY, const int* ldy, const int ldldY, const int ldPtrY,
    float** dZ, const int* ldz, const int ldldZ, const int ldPtrZ,
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
    const double zLayerMult
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx >= n) return;

    const Indices inds(idx, dim);

    const Grad grad(mat, inds, dim, ldMat, ldldMat, ldPtrMat);

    switch(idx / dim[6]){ 
    	case 0: dX[inds.page(ldPtrX)][inds.word(ldx, ldldX)] = grad.at(inds.col,      dim[1],          1, 0, ldMat[inds.page(ldldMat)]); break;
	case 1: dY[inds.page(ldPtrY)][inds.word(ldy, ldldY)] = grad.at(inds.row,      dim[0],          1, 0, 1                        ); break;
	case 2: dZ[inds.page(ldPtrZ)][inds.word(ldz, ldldZ)] = grad.at(inds.layerInd, dim[2], zLayerMult, 1, 0                        );
    }
}
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
extern "C" __global__ void batchGradients3d(
    const int n, 
    const float** mat, const int* ldMat, const int ldldMat, const int ldPtrMat,
    float** dX, const int* ldx, const int ldldX, const int ldPtrX,
    float** dY, const int* ldy, const int ldldY, const int ldPtrY,
    float** dZ, const int* ldz, const int ldldZ, const int ldPtrZ,
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
    const double zLayerMult
) {    
    batchGradients(    
        n, 
        mat, ldMat, ldldMat, ldPtrMat,
        dX, ldx, ldldX, ldPtrX,
        dY, ldy, ldldY, ldPtrY,
        dZ, ldz, ldldZ, ldPtrZ,
        dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
        zLayerMult
    );
}
extern "C" __global__ void batchGradients2d(
    const int n, 
    const float** mat, const int* ldMat, const int ldldMat, const int ldPtrMat,
    float** dX, const int* ldx, const int ldldX, const int ldPtrX,
    float** dY, const int* ldy, const int ldldY, const int ldPtrY,    
    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
    const double zLayerMult
) {

    batchGradients(    
        n, 
        mat, ldMat, ldldMat, ldPtrMat,
        dX, ldx, ldldX, ldPtrX,
        dY, ldy, ldldY, ldPtrY,
        nullptr, nullptr, -1, -1,
        dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
        zLayerMult
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * Provides the leading dimensions for the desired layer in the desired tensor.
 */
class XYLd {
private:
    const int* xyLd; ///< Pointer to leading dimension array.
    const int ldld;  ///< Stride for indexing layers.

public:
    /**
     * Constructor.
     * @param xyLd Pointer to base array holding the leading dimensions of all the layers.
     * @param ldld Leading dimension (stride) beteen columns of xyLd.
     */
    __device__ XYLd(const int* xyLd, int ldld) : xyLd(xyLd), ldld(ldld) {}

    /**
     * Indexing operator.
     * @param layer Layer index (e.g., z-slice).
     * @param tensor Tensor index (e.g., batch).
     * @return The XY leading dimension for the given tensor layer.
     */
    __device__ int operator()(int layer, int tensor) const {
        return xyLd[tensor * ldld + layer];
    }
};

class Vec {
private:
    double** data;
    const int xyInd0;
    const int ztInd0;
    const int xyStep;
    const int ztStep;

public:
    /**
     * Default constructor does nothing
     * @param data the data
     * @param xy0 Starting xy-coordinate index.
     * @param zt Starting zt-coordinate index.
     * @param dxy Step in xy-direction (0 or 1 usually).
     * @param strideZT Step in zt-direction (0 or 1 usually).
     */
    __device__ Vec(double** data, int xy0, int zt0, int strideXY, int strideZT): data(data), xyInd0(xy0), ztInd0(zt0), xyStep(strideXY), ztStep(strideZT) {}

    __device__ double& operator[](int i){
        return data[ztInd0 + i * ztStep][xyInd0 + i * xyStep];
    }
};

/**
 * Computes the neighborhood with a rolling sum.
 * @param r the radius of the nieghborhood.
 * @param numSteps The length of each Vec.
 * @param src The src vector whose some is to be taken.
 * @param dst where the sums are to be placed.
 */
__device__ void rollTheSum(const int r, const int numSteps, Vec& src, Vec& dst) {
    double rollingSum = 0;

    int m = min(r + 1, numSteps);

    for (int i = 0; i < m; i++) //loop 1: init 1st value
        rollingSum += src[i];

    dst[0] = rollingSum;//i = 0

    int i = 1;

    m = min(r + 1, numSteps - r);
    for (; i < m; i++) dst[i] = (rollingSum += src[i + r]); //loop 2: first section

    m = numSteps - r;
    for (; i < m; i++) dst[i] = (rollingSum += src[i + r] - src[i - r - 1]); //loop 3: mid section

    m = min(r + 1, numSteps);
    for(;i < m; i++) dst[i] = rollingSum; //loop 4: end section for big r

    for (; i < numSteps; i++) dst[i] = (rollingSum -= src[i - r - 1]); //loop 5: end section for small r
}

/**
 * CUDA kernel to compute a directional rolling sum over a 3D tensor.
 *
 * @param n Total number of elements to process.
 * @param srcData Source tensor in global memory.
 * @param xyLdSrc Leading dimensions for source in x/y.
 * @param ldldSrc Leading dimension stride for x/y.
 * @param ztLdSrc Leading dimension for z/t in source.
 * @param dstData Destination tensor in global memory.
 * @param xyLdDst Leading dimensions for destination in x/y.
 * @param ldldDst Leading dimension stride for x/y in destination.
 * @param ztLdDst Leading dimension for z/t in destination.
 * @param height Height of the tensor (Y).
 * @param width Width of the tensor (X).
 * @param depth Depth of the tensor (Z).
 * @param numSteps Total number of steps to compute.
 * @param r Radius of the summation window.
 */
extern "C" __global__ void neighborhoodSum3dKernelX(
    const int n,
    double** srcData, const int* xyLdSrc, const int ldldSrc, const int ztLdSrc,
    double** dstData, const int* xyLdDst, const int ldldDst, const int ztLdDst,

    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6

    const int numSteps,

    const int r
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return; // Out-of-bounds thread

    XYLd xySrcLd(xyLdSrc, ldldSrc), xyDstLd(xyLdDst, ldldDst);

    int row = idx % dim[0], absLayer = idx / dim[0], layer = absLayer % dim[2], tensor = absLayer/dim[2];

    Vec src(srcData, row, tensor * ztLdSrc + layer, xySrcLd(layer, tensor), 0),
        dst(dstData, row, tensor * ztLdDst + layer, xyDstLd(layer, tensor), 0);

    rollTheSum(r, numSteps, src, dst);
}

extern "C" __global__ void neighborhoodSum3dKernelY(
    const int n,
    double** srcData, const int* xyLdSrc, const int ldldSrc, const int ztLdSrc,
    double** dstData, const int* xyLdDst, const int ldldDst, const int ztLdDst,

    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6

    const int numSteps,

    const int r
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return; // Out-of-bounds thread

    XYLd xySrcLd(xyLdSrc, ldldSrc), xyDstLd(xyLdDst, ldldDst);

    int col = idx % dim[1], absLayer = idx/dim[1], layer = absLayer % dim[2], tensor = absLayer / dim[2];

    Vec src(srcData, col * xySrcLd(layer, tensor), tensor * ztLdSrc + layer, 1, 0),
        dst(dstData, col * xyDstLd(layer, tensor), tensor * ztLdDst + layer, 1, 0);

    rollTheSum(r, numSteps, src, dst);
}

extern "C" __global__ void neighborhoodSum3dKernelZ(
    const int n,
    double** srcData, const int* xyLdSrc, const int ldldSrc, const int ztLdSrc,
    double** dstData, const int* xyLdDst, const int ldldDst, const int ztLdDst,

    const int* dim, //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6

    const int numSteps,

    const int r,
    const int direction
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return; // Out-of-bounds thread

    XYLd xySrcLd(xyLdSrc, ldldSrc), xyDstLd(xyLdDst, ldldDst);

    int idxInLayer = idx % dim[4], row = idxInLayer % dim[0], col = idxInLayer/dim[0], tensor = idx / dim[4];

    Vec src(srcData, col * xySrcLd(0, tensor) + row, tensor * ztLdSrc, 0, 1),
        dst(dstData, col * xyDstLd(0, tensor) + row, tensor * ztLdDst, 0, 1);

    rollTheSum(r, numSteps, src, dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief CUDA kernel that performs elementwise computation:
 *        `dst = timesDst * dst + timesProduct * a * b`
 *
 * This kernel operates on 4D batched data (frames × depth × height × width), allowing
 * for strided memory access via pointer arrays and leading-dimension arrays. It reads
 * corresponding values from two inputs (`a` and `b`), multiplies them, scales the result,
 * and accumulates it with a scaled value from the destination (`dst`).
 *
 * @param n         Total number of elements to process (threads).
 * @param dst       Pointer array to output 2D slices (modifiable).
 * @param xyLdDst   Leading dimension array for `dst` (per slice).
 * @param ldldDst   Stride across `xyLdDst` for indexing slices.
 * @param ztLdDst   Stride across `dst` for frame × depth indexing.
 *
 * @param a         Pointer array to 2D slices of input A.
 * @param xyLdA     Leading dimension array for input A slices.
 * @param ldldA     Stride across `xyLdA` for indexing.
 * @param ztLdA     Stride across `a` for frame × depth indexing.
 *
 * @param b         Pointer to 1D flattened array of input B.
 * @param xyLdB     Leading dimension array for input B.
 * @param ldldB     Stride across `xyLdB` for indexing.
 * @param ztLdB     Stride across `b` for frame × depth indexing.
 *
 * @param dim       height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
 *
 * @param timesProduct Scalar multiplier for the product of `a` and `b`.
 *
 *
 * @param timesDst  Scalar multiplier applied to the existing value in `dst`.
 */
extern "C" __global__ void setEBEProductKernel(
    const int n,
    double** dst, const int* xyLdDst, const int ldldDst, const int ztLdDst,
    const float** a, const int* xyLdA, const int ldldA, const int ztLdA,
    const float** b, const int* xyLdB, const int ldldB, const int ztLdB,

    const int* dim //height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Indices ind(idx, dim);

    ind(dst, xyLdDst, ldldDst, ztLdDst) = ind(a, xyLdA, ldldA, ztLdA) * ind(b, xyLdB, ldldB, ztLdB);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief CUDA kernel that multiplies each element in a 4D dataset by a scalar.
 *
 * The dataset is organized as: frames × depth × height × width. Each 2D layer
 * is stored in column-major order and accessed via pointers with stride metadata.
 *
 * @param totalElements   Total number of elements to process.
 * @param pointersToLayers 2D array of pointers to 2D layers.
 * @param ldLayers        2D array of column strides per layer.
 * @param ldld            Leading dimension of ldLayers (stride across depth).
 * @param ldPtrs          Leading dimension of pointersToLayers (stride across frames).
 * @param dim The dimensions
 * @param scalar          Scalar to multiply each element by.
 */
extern "C" __global__ void multiplyScalarKernel(
    const int totalElements,
    float** pointersToLayers, const int* ldLayers, const int ldld, const int ldPtrs,
    const int* dim,
    const float scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    Indices mapper(idx, dim);

    mapper(pointersToLayers, ldLayers, ldld, ldPtrs) *= scalar;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Uses Kahan's method for more accurate mulitplication.
 */
__device__ double prod(double a, double b){
    double result = a*b;
    return result - ::fma(a, b, -result);
}

/**
 * Swap function for double values.
 *
 * @param a First value.
 * @param b Second value.
 */
__device__ inline void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}
/**
 * Checks if the two values are close to one another.
 * @param a The first value.
 * @param b The second value.
 * @param The tolerance.
 */
__device__ bool eq(double a, double b, double tol){
	return fabs(a - b) < tol;
}


/**
 * @class MaxAbs
 * @brief A utility class for tracking the argument corresponding to the maximum absolute value in a set of comparisons.
 *
 * This class is designed for use in CUDA device code and provides methods to update the tracked maximum
 * absolute value and retrieve the corresponding argument.
 */
class MaxAbs {
private:
    int arg; ///< The argument corresponding to the maximum absolute value.
    double val; ///< The maximum absolute value encountered so far.

public:
    /**
     * @brief Constructor for the MaxAbs class.
     *
     * Initializes the maximum absolute value and its corresponding argument.
     *
     * @param initVal The initial maximum absolute value.
     * @param initArg The initial argument corresponding to the maximum absolute value.
     */
    __device__ MaxAbs(int initArg, double initVal) : arg(initArg), val(fabs(initVal)) {}

     /**
     * @brief Updates the tracked maximum absolute value if the new value is greater.
     *
     * Compares the given value with the current maximum absolute value. If the new value is greater,
     * updates the maximum value and its corresponding index.
     *
     * @param candidateIndex The index associated with the new value.
     * @param candidateValue The new value to compare against the current maximum absolute value.
     */
    __device__ void challenge(int candidateIndex, double candidateValue) {
        double absoluteValue = fabs(candidateValue); // Compute the absolute value of the candidate value.
        if (absoluteValue > val) {             // Update if the candidate value is larger than the current maximum.
            val = absoluteValue;
            arg = candidateIndex;
        }
    }

    /**
     * @brief Retrieves the argument corresponding to the maximum absolute value.
     *
     * @return The argument corresponding to the maximum absolute value.
     */
    __device__ int getArg() {
        return arg;
    }

    /**
     * @brief Retrieves the absolute value at the argument.
     *
     * @return The maximum absolute value.
     */
    __device__ double getVal() {
        return val;
    }
};

/**
 * Represents a 3x3 symmetric matrix in column-major format.
 */
class Matrix3x3 {
private:
    double mat[3][3];
    double tolerance;

    /**
     * A value that is less than tolerance will be returned as 0.  Otherwise as itself.
     */
    __device__ double zeroBar(double maybeNear0){
        return eq(maybeNear0, 0, tolerance)? 0: maybeNear0;
    }
public:
    /**
     * Constructs a Matrix3x3 object.
     *
     * @param xx Element at (0,0).
     * @param xy Element at (0,1) and (1,0).
     * @param xz Element at (0,2) and (2,0).
     * @param yy Element at (1,1).
     * @param yz Element at (1,2) and (2,1).
     * @param zz Element at (2,2).
     * @param tol the tolerance.
     */
    __device__ explicit Matrix3x3(const double xx, const double xy, const double xz, const double yy, const double yz, const double zz, double tol) : tolerance(tol) {
        mat[0][0] = zeroBar(xx);
        mat[0][1] = mat[1][0] = zeroBar(xy);
        mat[0][2] = mat[2][0] = zeroBar(xz);
        mat[1][1] = zeroBar(yy);
        mat[1][2] = mat[2][1] = zeroBar(yz);
        mat[2][2] = zeroBar(zz);
    }



    /**
     * Computes the trace of the matrix.
     * @return The sum of the diagonal elements.
     */
    __device__ double trace() const {
        return mat[0][0] + mat[1][1] + mat[2][2];
    }

    /**
     * Computes the sum of 2x2 determinant minors of the matrix.
     * @return The sum of determinant minors.
     */
    __device__ double diagMinorSum() const {
        return mat[1][1]*mat[2][2] - mat[1][2]*mat[1][2] + mat[0][0]*mat[2][2] - mat[0][2]*mat[0][2] + mat[0][0]*mat[1][1] - mat[0][1]*mat[0][1];
    }

    /**
     * Computes the determinant of the matrix.
     * @return The determinant value.
     */
    __device__ double determinant() const {
        return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[1][2]) -
               mat[0][1] * (mat[0][1] * mat[2][2] - mat[1][2] * mat[0][2]) +
               mat[0][2] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
    }

    /**
     * Subtracts the val from each element on the diagnal of this matrix, changing this matrix.
     * @param val The value to be subtracted from each element of this matrix.
     */
    __device__ void subtractFromDiag(double val){
        mat[0][0] -= val; mat[1][1] -= val; mat[2][2] -= val;
    }

    /**
     * @brief Access an element in the matrix by row and column index.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at the specified row and column.
     */
    __device__ double& operator()(int row, int col) {
        return mat[row][col];
    }

    /**
     * @brief Access a const element in the matrix by row and column index.
     * @param row Row index.
     * @param col Column index.
     * @return Const reference to the element at the specified row and column.
     */
    __device__ double operator()(int row, int col) const {
        return mat[row][col];
    }

    /**
     * @brief Subtracts a scaled row from another row.
     * @param minuendInd Index of the row to be updated.
     * @param subtrahendInd Index of the row to subtract.
     * @param scale Scaling factor.
     * @param The value at this column will be set to 0.  Perform subtraction to the right.
     */
    __device__ void subtractRow(int minuendInd, int subtrahendInd, double scale, int startCol) {
        mat[minuendInd][startCol] = 0;
        for (int x = startCol + 1; x < 3; x++)
			mat[minuendInd][x] = ::fma(-scale, mat[subtrahendInd][x], mat[minuendInd][x]);


    }

    /**
     * @brief Swaps two rows of the matrix.
     * @param i First row index.
     * @param j Second row index.
     * @param startCol begin swaping with this column and proceed to the right.
     */
    __device__ void swapRows(int i, int j, int startCol) {
        for(int k = startCol; k < 3; k++) swap(mat[i][k], mat[j][k]);
    }

    /**
     * Scales the row so that the element at the startCol is one and every element after is times one over that element.
     * @param row the row to be scaled.
     * @startCol the column index of the first non zero element of the row.
     */
    __device__ void scaleRow(int row, int startCol){

	double inv = 1/mat[row][startCol];
    	mat[row][startCol] = 1;
    	for(int i = startCol + 1; i < 3; i++) mat[row][i] *= inv;

    }


    /**
     * @brief Perform row echelon work for a specific row and column.
     * @param row Current row index.
     * @param col Current column index.
     * @return True if a pivot was found, false otherwise.
     */
    __device__ bool reduceToRowEchelon(const int row, const int col) {

        MaxAbs maxPivot(row, fabs(mat[row][col]));

	for (int y = row + 1; y < 3; y++) maxPivot.challenge(y, mat[y][col]);

        if (maxPivot.getVal() <= tolerance) return false;

        if (maxPivot.getArg() != row) swapRows(maxPivot.getArg(), row, col);

        scaleRow(row, col);

        for (int y = row + 1; y < 3; y++)
			if(fabs(mat[y][col]) > tolerance) subtractRow(y, row, mat[y][col], col);

        return true;
    }

    /**
     * @brief Perform row echelon reduction on the matrix.
     * @return Number of free variables found during the reduction.
     */
    __device__ int rowEchelon() {
        int numFreeVariables = 0;
        int row = 0;

        for (int col = 0; col < 3; col++)
            if (reduceToRowEchelon(row, col)) row++;
            else numFreeVariables++;

        return numFreeVariables;
    }

    /**
     * Prints the matrix for debugging purposes using a single printf.
     */
    __device__ void print() {
        printf("\nMatrix:\n%f %f %f\n%f %f %f\n%f %f %f\n",
               mat[0][0], mat[0][1], mat[0][2],
               mat[1][0], mat[1][1], mat[1][2],
               mat[2][0], mat[2][1], mat[2][2]);
    }

};

/**
 * Represents an affine function y = ax + b.
 */
class Affine {
private:
    double a; /**< The slope of the line. */
    double b; /**< The y-intercept of the line. */

public:
    /**
     * Constructs an Affine function.
     * @param a The slope.
     * @param b The y-intercept.
     */
    __device__ Affine(double a, double b) : a(a), b(b) {}

    /**
     * Evaluates the function at a given x.
     * @param x The input value.
     * @return The corresponding y-value.
     */
    __device__ double operator()(double x) {
        return ::fma(a, x, b);
    }

    /**
     * Maps multiple x-values to y-values.
     * @param x1 First x-value.
     * @param x2 Second and thid x-values.
     * @param y Pointer to an array where results are stored.
     */
    __device__ void map(double x1, double x2And3, double* y) {
        y[0] = (*this)(x1);
        y[1] = y[2] = (*this)(x2And3);
    }

    /**
     * @return The slope of the function.
     */
    __device__ double getSlope(){
        return a;
    }

    /**
     * Prints the function parameters.
     */
    __device__ void print(){
        printf("a = %lf and b = %lf\n\n", a, b);
    }
};


/**
 * @class Vec
 * @brief A simple wrapper for a double array representing a 3D vector.
 */
class Vec3 {
private:
    double data[3];
    double tolerance;

    /**
     * Sorts an array in descending order.
     */
    __device__ void sortDescending() {
        if(data[0] < data[1]) swap(data[0], data[1]);
        if(data[0] < data[2]) swap(data[0], data[2]);
        if(data[1] < data[2]) swap(data[1], data[2]);
    }

    /**
     * Computes the real roots of a cubic equation and stores them in this vector.
     *
     * @param b Coefficient of x^2.
     * @param c Coefficient of x.
     * @param d Constant term.
     * @param eigenInd The index of the eigenvalue to be returned from this method.  0 for the largest eigenValue and 2 for the smallest.
     * @param val Output array to store roots.
     * @return The eigen value at the desired index.
     */
    __device__ void cubicRoot(const double b, const double c, const double d){

	double inv3 = 1.0/3;

	double nBInv3 = -b*inv3;

	double p = ::fma(nBInv3, b, c) * inv3;
	double q = ::fma(::fma(b/13.5, b, -c*inv3), b, d);

	if (p >= -1e-9) set(nBInv3, nBInv3, nBInv3);

	else{

	    Affine line(2 * sqrt(-p), nBInv3);

	    double arg = q/prod(line.getSlope(), p);

	    if(arg > 1 - 1e-6) line.map(1, -0.5, data);
	    else if(arg < -1 + 1e-6) line.map(-1, 0.5, data);
	    else {

	        double acosArg = acos(arg);

	        set(line(cos(acosArg * inv3)),
 	            line(cos(::fma(2.0, M_PI, acosArg) * inv3)),
	            line(cos(::fma(4.0, M_PI, acosArg) * inv3))
	    	);
	    }
	}
    }

public:
    /**
     * @brief Constructs a Vec object.
     * @param data Pointer to the double array (size 3) representing the vector.
     */
    __device__ Vec3(double tolerance): tolerance(tolerance){}

    /**
     * @brief Sets the components of the vector.
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     */
    __device__ void set(double x, double y, double z){
        data[0] = x; data[1] = y; data[2] = z;
    }

    /**
     * @brief Accesses a component of the vector using array-like indexing.
     * @param i The index of the component (0 for x, 1 for y, 2 for z).
     * @return A reference to the requested vector component.
     */
    __device__ double& operator[](int i) {
        return data[i];
    }

    /**
     * gets the element at the ith index.
     * @param i The index of the component (0 for x, 1 for y, 2 for z).
     * @return The element at the ith index.
     */
    __device__ double operator()(int i) const{
        return data[i];
    }

    /**
     * @brief Prints the components of the vector to the standard output.
     * The output format is "(x, y, z)".
     */
    __device__ void print() const {
        printf("(%f, %f, %f)\n", data[0], data[1], data[2]);
    }

    /**
     * @brief Checks if any of the vector's components are NaN (Not a Number).
     * @return True if at least one component is NaN, false otherwise.
     */
    __device__ bool hasNaN() const {
        return isnan(data[0]) || isnan(data[1]) || isnan(data[2]);
    }

        /**
     * @brief Calculates the squared length (magnitude) of the vector.
     * @return The squared length of the vector.
     */
    __device__ double lengthSquared() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }

    /**
     * @brief Calculates the length (magnitude) of the vector.
     * @return The length of the vector.
     */
    __device__ double length() const {
        return sqrtf(lengthSquared());
    }

    /**
     * @brief Normalizes the vector in-place, setting its length to 1.
     * Also, if the y value is less than 0, the vector is mulitplied by -1;
     * If the vector's length is zero, it remains unchanged.
     */
    __device__ void normalize() {
        double len = length();
        if (len > 0.0f) {
            double invLen = 1.0 / len;
            data[0] *= invLen;
            data[1] *= invLen;
            data[2] *= invLen;
            if(data[1] < 0 || (data[1] == 0 && (data[0] < 0 || (data[0] == 0 && data[2] < 0))))
                for(int i = 0; i < 3; i++) data[i] *= -1;
        }

    }

    /**
     * @brief Computes an eigenvector based on the number of free variables after row reduction
     * of a matrix (A - lambda * I), where lambda is an eigenvalue.
     *
     * This method sets the components of this Vec3 object to represent the eigenvector.
     * The eigenvector is determined based on the number of free variables found during
     * the row echelon form reduction of the matrix and the index of the eigenvalue being considered.
     *
     * @param mat The 3x3 matrix (A - lambda * I) in row-major format after row reduction.
     * @param freeVariables The number of free variables resulting from the row reduction.
     * @param eigenInd The index of the eigenvalue (0, 1, or 2) for which the eigenvector is being computed.
     */
    __device__ void setEVec(const Matrix3x3& mat, int freeVariables, int eigenInd) {
        double smTol = 1e-6;

        if(freeVariables == 1){
            if (fabs(mat(0, 0)) <= smTol) set(1, 0, 0);
            else if (fabs(mat(1, 1)) <= smTol) set(-mat(0, 1) / mat(0, 0), 1, 0);
            else {
                data[2] = 1;
                data[1] = -mat(1, 2) / mat(1, 1);
                data[0] = (-mat(0, 2) - mat(0, 1) * data[1]) / mat(0, 0);
            }
            normalize();
        } else set(NAN, NAN, NAN);
    }

    /**
     * The azimuthal angle of this vector.
     */
    __device__ float azimuth(){
        if(isnan(data[0]) || data[0]*data[0] + data[1]*data[1] <= tolerance) return nan("");
       return fmod(atan2(data[1], data[0]) + M_PI, M_PI);
    }

    /**
     * The zenith angle of this vector.
     */
    __device__ float zenith(){
        if(isnan(data[0]) || lengthSquared() <= tolerance) return nan("");
        else if(data[2] >= 1 - tolerance) return 0;
        else if(data[2] <= tolerance - 1) return M_PI;
        else return acos(fabs(data[2]));
    }

    /**
     * Finds the eigenvalues.
     *@param mat The matrix for whom the eigenvalues are desired.
     */
    __device__ void setEVal(const Matrix3x3& mat){
        cubicRoot(-mat.trace(), mat.diagMinorSum(), -mat.determinant());
        sortDescending();
    }

    /**
     * The multiplicity at the requested index.
     * @param the index of the desired multiplicity.
     */
    __device__ int multiplicity(int ind){
        return (data[0] == data[ind]) + (data[1] == data[ind]) + (data[2] == data[ind]) - 1;
    }

    /**
     * Copies these values to the desired location.
     */
    __device__ void writeTo(float* dst){
     	for(int i = 0; i < 3; i++) dst[i] = (float)data[i];
    }

    __device__ double coherence(){
        if(isnan(data[0])) return 0;
        return data[0] <=  tolerance ? 0 : (data[1] - data[2]) / (data[1] + data[2]);
    }

};



/**
 * CUDA Kernel to compute eigenvalues and eigenvectors of a batch of 3x3 symmetric matrices.
 *
 * @param n Total number of input elements before downsampling.
 * @param xx Array of pointers to the xx components of each height x width slice (row is depth and col is frame.).
 * @param ldxx Array of leading dimensions for the xx components of each slice (size: depth * batchSize).
 * @param ldldxx Leading dimension of the ldxx array (stride between leading dimensions in memory).
 * @param ldPtrxx Leading dimension of the xx pointer array (stride between pointers in memory).
 * @param xy Array of pointers to the xy components of each height x width slice (organized by depth then batch).
 * @param ldxy Array of leading dimensions for the xy components of each slice (size: depth * batchSize).
 * @param ldldxy Leading dimension of the ldxy array.
 * @param ldPtrxy Leading dimension of the xy pointer array.
 * @param xz Array of pointers to the xz components of each height x width slice (organized by depth then batch).
 * @param ldxz Array of leading dimensions for the xz components of each slice (size: depth * batchSize).
 * @param ldldxz Leading dimension of the ldxz array.
 * @param ldPtrxz Leading dimension of the xz pointer array.
 * @param yy Array of pointers to the yy components of each height x width slice (organized by depth then batch).
 * @param ldyy Array of leading dimensions for the yy components of each slice (size: depth * batchSize).
 * @param ldldyy Leading dimension of the ldyy array.
 * @param ldPtryy Leading dimension of the yy pointer array.
 * @param yz Array of pointers to the yz components of each height x width slice (organized by depth then batch).
 * @param ldyz Array of leading dimensions for the yz components of each slice (size: depth * batchSize).
 * @param ldldyz Leading dimension of the ldyz array.
 * @param ldPtryz Leading dimension of the yz pointer array.
 * @param zz Array of pointers to the zz components of each height x width slice (organized by depth then batch).
 * @param ldzz Array of leading dimensions for the zz components of each slice (size: depth * batchSize).
 * @param ldldzz Leading dimension of the ldzz array.
 * @param ldPtrzz Leading dimension of the zz pointer array.

 * @param dim  height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6

 * @param tolerance Tolerance for floating-point comparisons.
 * @param zenith where the zenith angles, between 0 and pi, will be stored.
 * @param azimuthal where the Azimuthal angles, between 0 and pi, will be stored.
 * @param coherence The coherence of the vector will be stored here.  Note, if this value is negaitvie then the vector at this index is normal to the plane with a coherence equal to the absolute value of the coherence stored here.
 */
extern "C" __global__ void eigenBatch3dKernel(
    const int n,

    const double** xx, const int* ldxx, const int ldldxx, const int ldPtrxx,
    const double** xy, const int* ldxy, const int ldldxy, const int ldPtrxy,
    const double** xz, const int* ldxz, const int ldldxz, const int ldPtrxz,
    const double** yy, const int* ldyy, const int ldldyy, const int ldPtryy,
    const double** yz, const int* ldyz, const int ldldyz, const int ldPtryz,
    const double** zz, const int* ldzz, const int ldldzz, const int ldPtrzz,

    float** eVecs, const int* ldEVec, const int ldldEVec, const int ldPtrEVec,
    float** coherence, const int* ldCoh, const int ldldCoh, const int ldPtrCoh,
    float** azimuthal, const int* ldAzi, const int ldldAzi, const int ldPtrAzi,
    float** zenith, const int* ldZen, const int ldldZen, const int ldPtrZen,

    const int* dim,

    const int downSampleFactorXY, const int downSampleFactorZ, int eigenInd,
    const double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    Indices src(idx, dim, downSampleFactorXY, downSampleFactorZ);
    Indices dst(idx, dim, 1, 1);

    Matrix3x3 mat(
    	src(xx, ldxx, ldldxx, ldPtrxx), src(xy, ldxy, ldldxy, ldPtrxy), src(xz, ldxz, ldldxz, ldPtrxz),
                                        src(yy, ldyy, ldldyy, ldPtryy), src(yz, ldyz, ldldyz, ldPtryz),
    					                                src(zz, ldzz, ldldzz, ldPtrzz),
        tolerance
    );

    Vec3 eVals(tolerance);
    eVals.setEVal(mat);

	bool ortho = eq(eVals(1), eVals(2), tolerance) && !eq(eVals(0), (eVals(1) +eVals(2))/2, tolerance);
	if(ortho) eigenInd = 0;
    dst(coherence, ldCoh, ldldCoh, ldPtrCoh) = (ortho?-1:1)*(float)eVals.coherence();


    mat.subtractFromDiag(eVals[eigenInd]);

    Vec3 vec(1e-5);

    vec.setEVec(mat, mat.rowEchelon(), eigenInd);

    vec.writeTo(eVecs[dst.page(ldPtrEVec)] + dst.col * ldEVec[dst.page(ldldEVec)] + dst.row * 3);
    dst(azimuthal, ldAzi, ldldAzi, ldPtrAzi) = vec.azimuth();
    dst(zenith, ldZen, ldldZen, ldPtrZen) = vec.zenith();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * @class Matrix2x2
 * @brief Represents a 2x2 symmetric matrix.
 */
class Matrix2x2 {
private:
    double mat[2][2];  ///< 2x2 matrix storage.
    double tolerance;  ///< Tolerance for zero checks.

    __device__ double zeroBar(double maybeNear0) {
        return fabs(maybeNear0) <= tolerance ? 0 : maybeNear0;
    }

public:
    __device__ explicit Matrix2x2(const double xx, const double xy, const double yy, double tol) : tolerance(tol) {
        mat[0][0] = zeroBar(xx);
        mat[0][1] = mat[1][0] = zeroBar(xy);
        mat[1][1] = zeroBar(yy);
    }

    __device__ double trace() const { return mat[0][0] + mat[1][1]; }
    __device__ double determinant() const { return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]; }
    __device__ double operator()(int row, int col) const { return mat[row][col]; }

    /**
     * @brief Prints the contents of the Matrix2x2 object.
     * For debugging purposes.
     */
    __device__ void print() const {
        printf("Matrix2x2 Debug Info (Thread %d)\n", threadIdx.x + blockIdx.x * blockDim.x);
        printf("  [ %.6f  %.6f ]\n", mat[0][0], mat[0][1]);
        printf("  [ %.6f  %.6f ]\n", mat[1][0], mat[1][1]);
        printf("  Tolerance: %.6f\n", tolerance);
        printf("  Trace: %.6f\n", trace());
        printf("  Determinant: %.6f\n", determinant());
        printf("\n");
    }
};

/**
 * https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
 * @class Vec
 * @brief A simple wrapper for a double array representing a 2D vector.
 */
class Vec2 {
private:
    double data[2];
    double tolerance;

    __device__ void sortDescending() {
        if (data[0] < data[1]) swap(data[0], data[1]);
    }

public:
    __device__ Vec2(double tol) : tolerance(tol) { data[0] = 0; data[1] = 0; }
    __device__ void set(double x, double y) { data[0] = x; data[1] = y; }
    __device__ double& operator[](int i) { return data[i]; }
    __device__ double operator()(int i) const { return data[i]; }

    __device__ double lengthSquared() const {return data[0] * data[0] + data[1] * data[1];}
    __device__ double length() const { return sqrt(lengthSquared()); }

    __device__ void normalize() {
        double len = length();
        if (len > tolerance) {
            double invLen = 1.0 / len;
            data[0] *= invLen; data[1] *= invLen;
            if (data[1] < 0 || (fabs(data[1]) <= tolerance && data[0] < 0)) {
                data[0] *= -1; data[1] *= -1;
            }
        }
    }

    __device__ void setEVec(const Matrix2x2& mat, const double eVal, int vecInd) {
        if(fabs(mat(1, 0)) > tolerance) set(eVal - mat(1, 1), mat(1, 0));
        else if(fabs(mat(0, 1)) > tolerance) set(mat(0, 1), eVal - mat(0, 0));
        else if(vecInd) set(1, 0);
        else set(0, 1);

        normalize();
    }

    __device__ float angle() const {
        if (lengthSquared() <= 1e-6) return NAN;
        float angle = atan2(data[1], data[0]);
        if (angle < 0.0f) angle += M_PI;
        return angle;
    }

    __device__ void setEVal(const Matrix2x2& mat) {
        double t = mat.trace(), d = mat.determinant();
        double x = t/2, y = sqrt(max(t*t/4-d, (double)0));
        data[0] = x + y;
        data[1] = x - y;
    }

    __device__ double coherence() const {
        return (data[0] <= tolerance) ? 0 : (float)((data[0] - data[1]) / (data[0] + data[1]));
    }

    __device__ void writeTo(float* dst){
        dst[0] = (float)data[0];
        dst[1] = (float)data[1];
    }

    /**
     * @brief Prints pertinent debugging information for the Vec object.
     * Includes vector components and results of key methods.
     */
    __device__ void print() const {
        printf("Vec Debug Info (Thread %d)\n", threadIdx.x + blockIdx.x * blockDim.x);
        printf("  Components: [x=%.6f, y=%.6f]\n", data[0], data[1]);
        printf("  Tolerance: %.6f\n", tolerance);
        printf("  Length: %.6f\n", length());
        printf("\n");
    }
};

/**
 * @brief CUDA Kernel to compute eigenvalues/vectors of 2x2 matrices with downsampling.
 *
 * @param n_ds Total number of *downsampled* elements (pixels * frames).
 * @param xx, xy, yy Input structure tensor components (arrays of pointers).
 * @param ldxx, ldxy, ldyy Leading dimensions (heights) for input tensors.
 * @param ldldxx, ... (unused, but kept for consistency with 3D example if needed).
 * @param ldPtrxx, ... (unused, but kept for consistency).
 * @param eVecs, coherence, angle Output arrays (arrays of pointers).
 * @param ldEVec, ldCoh, ldAng Leading dimensions (heights) for output tensors.
 * @param ldldEVec, ... (unused).
 * @param ldPtrEVec, ... (unused).
 * @param dim Original dimensions {height, width, numFrames, imageSize}.
 * @param dsDim Downsampled dimensions {h_ds, w_ds, numFrames, imageSize_ds}.
 * @param downSampleFactor Downsampling factor (e.g., 2, 4).
 * @param eigenInd Index of eigenvector (0 for primary, 1 for secondary).
 * @param tolerance Floating-point tolerance.
 */
extern "C" __global__ void eigenBatch2dKernel(
    const int n_ds, // Use n_ds to indicate it's the downsampled size

    const double** xx, const int* ldxx, const int ldldxx, const int ldPtrxx,
    const double** xy, const int* ldxy, const int ldldxy, const int ldPtrxy,
    const double** yy, const int* ldyy, const int ldldyy, const int ldPtryy,

    float** eVecs, const int* ldEVec, const int ldldEVec, const int ldPtrEVec,
    float** coherence, const int* ldCoh, const int ldldCoh, const int ldPtrCoh,
    float** angle, const int* ldAng, const int ldldAng, const int ldPtrAng,

    const int* dim,

    const int downSampleFactor, // Add downsampling factor
    const int eigenInd,
    const double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_ds) return; // Check bounds against downsampled size

    Indices src(idx, dim, downSampleFactor);

    const Matrix2x2 mat(
    	src(xx, ldxx, ldldxx, ldPtrxx), src(xy, ldxy, ldldxy, ldPtrxy),
                                        src(yy, ldyy, ldldyy, ldPtryy),
        tolerance
    );

    Vec2 eVals(1e-5);
    eVals.setEVal(mat);

    Indices dst(idx, dim, 1);

    dst(coherence, ldCoh, ldldCoh, ldPtrCoh) = (float)eVals.coherence();

    Vec2 vec(tolerance);
    vec.setEVec(mat, eVals(eigenInd), eigenInd);

    vec.writeTo(eVecs[dst.page(ldPtrEVec)] + dst.col * ldEVec[dst.page(ldldEVec)] + 2 * dst.row);
    dst(angle, ldAng, ldldAng, ldPtrAng) = vec.angle();
}
