/**
 * Represents a pixel in a 2D matrix and provides utility methods for kernel operations.
 * The pixel can move along a row or a column in the matrix and provides access to its values in two matrices.
 */
class Pixel {
private:
    int pos;
    int vecIndex;
    int height;
    bool isRowTraversal;      /**< Whether the pixel moves along the row (true) or column (false). */
    float* sourceMat;     /**< Pointer to the source matrix. */
    float* targetMat;     /**< Pointer to the target matrix. */
    int targetInc;

    /**
     * Computes the linear index for accessing the value in a 2D matrix stored in a row-major format.
     *
     * @param pos The position along the row or column.
     * @param vecIndex The index of the row or column.
     * @return The linear index in the matrix.
     */
    __device__ int index(int pos, int vecIndex) {
        return isRowTraversal ? pos * height + vecIndex: vecIndex * height + pos;
    }

public:
    /**
     * Constructor for the Pixel class.
     * Initializes the pixel's position, movement direction, and matrix references.
     *
     * @param vecIndex The index of the row or column.
     * @param isRowTraversal If true, the pixel moves along the row; otherwise, it moves along the column.
     * @param sourceMat Pointer to the source matrix.
     * @param targetMat Pointer to the target matrix.
     */
    __device__ Pixel(int vecIndex, bool isRowTraversal, float* sourceMat, float* targetMat, int height, int targetInc) 
        : pos(0), vecIndex(vecIndex), isRowTraversal(isRowTraversal), 
          sourceMat(sourceMat), targetMat(targetMat), height(height), targetInc(targetInc) {}

    /**
     * Retrieves the value from the source matrix at the current position.
     *
     * @return The value from the source matrix at the current position.
     */
    __device__ float sourceValue() {
        return sourceValue(0);
    }

    /**
     * Retrieves the value from the source matrix with an offset applied.
     *
     * @param offset The offset to apply to the current position.
     * @return The value from the source matrix at the offset position.
     */
    __device__ float sourceValue(int offset) {
        return sourceMat[index(pos + offset, vecIndex)];
    }

    /**
     * Retrieves a reference to the value in the target matrix at the current position.
     *
     * @return Reference to the value in the target matrix at the current position.
     */
    __device__ float& targetValue() {
        return targetValue(0);
    }

    /**
     * Retrieves a reference to the value in the target matrix with an offset applied.
     *
     * @param offset The offset to apply to the current position.
     * @return Reference to the value in the target matrix at the offset position.
     */
    __device__ float& targetValue(int offset) {        
        return targetMat[index(pos + offset, vecIndex) * targetInc];
    }
    
    /**
     * Moves the pixel forward one.
     */
     __device__ void move(){
     	++pos;
     }
};

/**
 * CUDA kernel for summing up nearby elements in a row or column of a 2D matrix.
 * The kernel computes a rolling sum over a neighborhood window.
 *
 * @param sourceMat Pointer to the source matrix stored in GPU memory.
 * @param vecLength The number of elements in each row or column.
 * @param targetMat Pointer to the target matrix where the results will be stored.
 * @param neighborhoodSize The size of the neighborhood for summation.
 * @param numVecs The number of rows or columns in the matrix.
 * @param isRowTraversal If true, the kernel operates on rows; otherwise, it operates on columns.
 * @param targetInc The stride for accessing values in the target matrix.
 */
extern "C" __global__ void neighborhoodSumKernel(
    int numVecs, float* sourceMat, 
    int vecLength, 
    float* targetMat, 
    int targetInc, 
    bool isRowTraversal, 
    int neighborhoodSize
    ) {

    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;   

    if (threadIndex >= numVecs) return;
    
    Pixel pixel(threadIndex, isRowTraversal, sourceMat, targetMat, isRowTraversal? numVecs: vecLength, targetInc);

    pixel.targetValue() = 0;    
    for (int i = 0; i <= neighborhoodSize; i++)
        pixel.targetValue() += pixel.sourceValue(i);

    int i = 1;
    for (; i <= neighborhoodSize; i++) {
        pixel.move();
        pixel.targetValue() = pixel.targetValue(-1) + pixel.sourceValue(neighborhoodSize);
    }

    // Compute the rolling sum for the middle region
    for (; i < vecLength - neighborhoodSize; i++) {
        pixel.move();
        pixel.targetValue() = pixel.targetValue(-1) - pixel.sourceValue(-neighborhoodSize - 1) + pixel.sourceValue(neighborhoodSize);
    }

    // Compute the rolling sum for the final region
    for (; i < vecLength; i++) {
        pixel.move();
        pixel.targetValue() = pixel.targetValue(-1) - pixel.sourceValue(-neighborhoodSize - 1);
    }
}

