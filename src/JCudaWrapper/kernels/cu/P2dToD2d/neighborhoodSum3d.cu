
enum Direction {
    X = 0,
    Y = 1,
    Z = 2
};


class PixelCursor {
private:
    double** data;
    int xyInd;
    int ztInd;
    int xyStep;
    int ztStep;

public:
    /**
     * Default constructor does nothing
     */
    __device__ PixelCursor(){}

    /**
     * Sets all internal state for this pixel cursor.
     *
     * @param data Pointer to matrix data (array of pointers to rows or layers).
     * @param xy Starting xy-coordinate index.
     * @param zt Starting zt-coordinate index.
     * @param dxy Step in xy-direction (0 or 1 usually).
     * @param dzt Step in zt-direction (0 or 1 usually).
     */
    __device__ void setAll(double** data, int xy, int zt, int dxy, int dzt) {
        this->data = data;
        this->xyInd = xy;
        this->ztInd = zt;
        this->xyStep = dxy;
        this->ztStep = dzt;
    }

    /** @return Current XY index. */
    __device__ int getXY() const { return xyInd; }

    /** @return Current ZT index. */
    __device__ int getZT() const { return ztInd; }

    /** Move the cursor by one step in xy and zt directions. */
    __device__ void move() { xyInd += xyStep; ztInd += ztStep; }

    /**
     * Get the value at the current position, with optional offset.
     * @param offset Offset from current position in the scan direction.
     * @return Value at the offset location.
     */
    __device__ double get(int offset = 0) const {
        return data[ztInd + offset * ztStep][xyInd + offset * xyStep];
    }

    /**
     * Set the value at the current position, with optional offset.
     * @param val Value to set.
     * @param offset Offset from current position in the scan direction.
     */
    __device__ void set(double val, int offset = 0) {
        data[ztInd + offset * ztStep][xyInd + offset * xyStep] = val;
    }
};

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
 * @param direction Direction of the summation: 0 = row, 1 = column, 2 = depth.
 */
extern "C" __global__ void neighborhoodSum3dKernel(
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

    PixelCursor src, dst;

    switch (direction) {
        case X: {
            int row = idx % dim[0], absLayer = idx / dim[0], layer = absLayer % dim[2], tensor = absLayer/dim[2];
            
            src.setAll(srcData, row, tensor * ztLdSrc + layer, xySrcLd(layer, tensor), 0);
            dst.setAll(dstData, row, tensor * ztLdDst + layer, xyDstLd(layer, tensor), 0);
            
        break;}  // Row-wise
        case Y: {
            int col = idx % dim[1], absLayer = idx/dim[1], layer = absLayer % dim[2], tensor = absLayer / dim[2];

            src.setAll(srcData, col * xySrcLd(layer, tensor), tensor * ztLdSrc + layer, 1, 0);
            dst.setAll(dstData, col * xyDstLd(layer, tensor), tensor * ztLdDst + layer, 1, 0);
	    
        break; }// Column-wise
        case Z: {
            int idxInLayer = idx % dim[4], row = idxInLayer % dim[0], col = idxInLayer/dim[0], tensor = idx / dim[4];
            
            src.setAll(srcData, col * xySrcLd(0, tensor) + row, tensor * ztLdSrc, 0, 1);
            dst.setAll(dstData, col * xyDstLd(0, tensor) + row, tensor * ztLdDst, 0, 1);            
        }//depth-wise
    }

    double rollingSum = 0;

    int m = min(r + 1, numSteps);

    for (int i = 0; i < m; i++) //loop 1: init 1st value
        rollingSum += src.get(i);
    
    dst.set(rollingSum);

    int i = 1;
    m = min(r + 1, numSteps - r); //loop 2: first section
    for (; i < m; i++) {
        src.move(); 
        dst.move();
        dst.set(rollingSum += src.get(r));
    }
    m = numSteps - r;
    for (; i < m; i++) { //loop 3: mid section
        src.move();
        dst.move();
        dst.set(rollingSum += src.get(r) - src.get(-r - 1));
    }
    m = min(r + 1, numSteps);
    for(;i < m; i++){//loop 4: end section for big r
        dst.move();
        dst.set(rollingSum);
    }
    for (; i < numSteps; i++) { //loop 5: end section for small r
	    src.move();
	    dst.move();
        dst.set(rollingSum -= src.get(-r - 1));
    }
}

