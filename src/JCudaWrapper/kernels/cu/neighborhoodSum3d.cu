/**
 * Represents a position in a 2D or 3D matrix and provides utility methods for neighborhood operations.
 * The class abstracts pixel movement along a defined direction (row, column, or depth) and allows 
 * efficient access to source and target matrices during CUDA kernel execution.
 */
class Pixel {
private:
    double* sourceMat;   /**< Pointer to the source matrix. */
    double* targetMat;   /**< Pointer to the target matrix. */
    int stepSize;        /**< Step size for moving along the specified direction. */
    int toInc;           /**< Increment multiplier for accessing the target matrix. */

public:
    /**
     * Constructs a Pixel instance.
     *
     * @param sourceMat Pointer to the source matrix.
     * @param targetMat Pointer to the target matrix.
     * @param stepSize The step size for moving along a direction in the matrix.
     * @param toInc The stride multiplier for accessing the target matrix.
     */
    __device__ Pixel(double* sourceMat, double* targetMat, int stepSize, int toInc)
        : sourceMat(sourceMat), targetMat(targetMat), stepSize(stepSize), toInc(toInc) {}

    /**
     * Retrieves the value from the source matrix at the current position plus an offset.
     *
     * @param offset The offset to apply (in units of `stepSize`).
     * @return The value from the source matrix.
     */
    __device__ double sourceValue(int offset = 0) const {
        return sourceMat[offset * stepSize];
    }

    /**
     * Accesses the value in the target matrix at the current position plus an offset.
     *
     * @param offset The offset to apply (in units of `stepSize`).
     * @return Reference to the value in the target matrix.
     */
    __device__ double& targetValue(int offset = 0) {
        return targetMat[offset * stepSize * toInc];
    }

    /**
     * Advances the pixel to the next position along the direction.
     */
    __device__ void move() {
        sourceMat += stepSize;
        targetMat += stepSize * toInc;
    }
};

/**
 * CUDA kernel for computing a rolling neighborhood sum over a 2D or 3D matrix along a specified direction.
 *
 * @param sourceMat Pointer to the source matrix in global memory.
 * @param targetMat Pointer to the target matrix in global memory.
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param depth The depth of the matrix (3rd dimension).
 * @param toInc The stride multiplier for accessing the target matrix.
 * @param neighborhoodSize The size of the neighborhood window for summation.
 * @param dir Direction of operation: 0 (row), 1 (column), or 2 (depth).
 */
extern "C" __global__ void neighborhoodSum3dKernel(
    int n,
    double* sourceMat,
    double* targetMat,
    int height, int width, int depth,
    int toInc,
    int neighborhoodSize,
    int dir
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return; // Out-of-bounds thread

    // Initialize starting position and step sizes
    int startIdx, stepSize, numSteps;
    switch (dir) {
        case 0: // Row-wise
            startIdx = (idx % height) + (idx / height) * (height * width);
            stepSize = height;
            numSteps = width;
            break;
        case 1: // Column-wise
            startIdx = idx * height;
            stepSize = 1;
            numSteps = height;
            break;
        case 2: // Depth-wise
            startIdx = idx;
            stepSize = height * width;
            numSteps = depth;
    }

    Pixel pixel(sourceMat + startIdx, targetMat + startIdx * toInc, stepSize, toInc);

    int rollingSum = 0;
    for (int i = 0; i <= neighborhoodSize; i++)
        rollingSum += pixel.sourceValue(i);
    
    pixel.targetValue() = rollingSum;

    int i = 1;
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

