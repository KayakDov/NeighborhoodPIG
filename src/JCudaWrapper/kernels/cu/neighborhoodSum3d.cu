/**
 * Represents a position in a 2D or 3D matrix and provides utility methods for neighborhood operations.
 * The class abstracts pixel movement along a defined direction (row, column, or depth) and allows 
 * efficient access to source and target matrices during CUDA kernel execution.
 */
class Pixel {
private:
    double* srcMat;   /**< Pointer to the source matrix. */
    const int srcStride;        /**< Step size for moving along the specified direction. */
    double* dstMat;   /**< Pointer to the target matrix. */    
    const int dstStride;           /**< Increment multiplier for accessing the target matrix. */

public:
    /**
     * Constructs a Pixel instance.
     *
     * @param srcMat Pointer to the source matrix.
     * @param dstMat Pointer to the target matrix.
     * @param srcStride The step size for moving along a direction in the matrix.
     * @param dstStride The stride multiplier for accessing the target matrix.
     */
    __device__ Pixel(double* srcMat, double* dstMat, const int srcStride, const int dstStride)
        : srcMat(srcMat), dstMat(dstMat), srcStride(srcStride), dstStride(dstStride) {}

    /**
     * Retrieves the value from the source matrix at the current position plus an offset.
     *
     * @param offset The offset to apply (in units of `srcStride`).
     * @return The value from the source matrix.
     */
    __device__ double sourceValue(int offset = 0) const {
        return srcMat[offset * srcStride];
    }

    /**
     * Accesses the value in the target matrix at the current position plus an offset.
     *
     * @param offset The offset to apply (in units of `srcStride`).
     * @return Reference to the value in the target matrix.
     */
    __device__ double& targetValue(int offset = 0) {
        return dstMat[offset * dstStride];
    }

    /**
     * Advances the pixel to the next position along the direction.
     */
    __device__ void move() {
        srcMat += srcStride;
        dstMat += dstStride;
    }
};

/**
 * CUDA kernel for computing a rolling neighborhood sum over a 2D or 3D matrix along a specified direction.
 *
 * @param srcMat Pointer to the source matrix in global memory.
 * @param dstMat Pointer to the target matrix in global memory.
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param ldDst If the destination is a matrix, then this is the leading dimension.  If it is a vector, then this is the increment.
 * @param depth The depth of the matrix (3rd dimension).
 * @param dstStride The stride multiplier for accessing the target matrix.
 * @param srcStride The size of each step in the 1d array to move through the relivant tensor dimension.
 * @param numSteps The number of steps to be taken in the desired dimension.
 * @param neighborhoodSize The size of the neighborhood window for summation.
 * @param dir Direction of operation: 0 (row), 1 (column), or 2 (depth).
 */
extern "C" __global__ void neighborhoodSum3dKernel(
    const int n,
    double* srcMat,
    double* dstMat,
    const int height, const int width, const int depth, const int ldSrc, const int ldDst,
    const int srcStride, const int dstStride, const int numSteps,
    const int neighborhoodSizeXY, const int neighborhoodSizeZ,
    const int dir,
    const bool dstIsMatrix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return; // Out-of-bounds thread
    
    // Initialize starting position and step sizes
    int srcStart, dstStart;
    int row = idx % height, col = idx / height;
    switch (dir) {
        case 0: 
            srcStart = row +  col * width * ldSrc;
            dstStart = dstIsMatrix ? row +  col * width * ldDst :  (row + col * height * width) * ldDst;
        break;  // Row-wise        
        case 1: 
            srcStart = idx * ldSrc;
	    dstStart = dstIsMatrix ? idx * ldDst : idx * height * ldDst;
        break; // Column-wise
        case 2: 
            int wd = width * depth, hw = height*width;
            srcStart = col*ldSrc + row + idx/hw *ldSrc * wd;
            dstStart = dstIsMatrix ?  col*ldDst + row + idx/hw * ldDst * wd: (col*height + row + idx/hw * height * wd) * ldDst;
        //depth-wise
    }

    Pixel pixel(srcMat + srcStart, dstMat + dstStart, srcStride, dstStride);

    double rollingSum = 0;
    for (int i = 0; i <= neighborhoodSize; i++)
        rollingSum += pixel.sourceValue(i);
    
    pixel.targetValue() = rollingSum;

    int i = 1;
    
    for (; i <= neighborhoodSize; i++) {
        pixel.move();
        rollingSum += pixel.sourceValue(neighborhoodSize);
        pixel.targetValue() = rollingSum;
    }
    
    for (; i < numSteps - neighborhoodSize; i++) {
        pixel.move();
        rollingSum += pixel.sourceValue(neighborhoodSize) - pixel.sourceValue(-neighborhoodSize - 1);
        pixel.targetValue() = rollingSum;
    }

    // Compute the rolling sum for the final region
    for (; i < numSteps; i++) {
        pixel.move();
        rollingSum -= pixel.sourceValue(-neighborhoodSize - 1);
        pixel.targetValue() = rollingSum;
    }
}

