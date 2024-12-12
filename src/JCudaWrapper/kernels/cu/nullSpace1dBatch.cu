#include <cstdio>

// Utility function for swapping two doubles
__device__ void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

/**
 * Class representing a single row in a column-major matrix.
 */
class Row {
private:
    double* data; /**< Pointer to the beginning of the row's data. */
    int width;    /**< Width of the matrix. */
    int ld;       /**< Leading dimension of the matrix. */

public:
    /**
     * Constructor for the Row class.
     * @param data Pointer to the row data (column-major format).
     * @param width Width of the matrix.
     * @param ld Leading dimension of the matrix.
     */
    __device__ Row(double* data, int width, int ld) : data(data), width(width), ld(ld) {}

    /**
     * Accessor for elements in the row.
     * @param col The column index of the desired element.
     * @return Reference to the element at the given column index.
     */
    __device__ double& operator()(int col) {
        return data[col * ld];
    }

    /**
     * Performs an in-place row subtraction: this_row -= timesOther * other_row.
     * @param other The row to subtract from this row.
     * @param timesOther The scalar multiplier for the other row.
     */
    __device__ void subtract(Row& other, double timesOther) {
        for (int i = 0; i < width; i++)
            (*this)(i) -= timesOther * other(i);
    }

    /**
     * Swaps the content of this row with another row.
     * @param other The row to swap with.
     */
    __device__ void swap(Row& other) {
        for (int i = 0; i < width; i++)
            ::swap((*this)(i), other(i));
    }
};

/**
 * Class representing a matrix in column-major format.
 */
class Matrix {
private:
    double* data; /**< Pointer to the matrix data. */
    int width;    /**< Width of the matrix. */
    int ld;       /**< Leading dimension of the matrix. */

public:
    /**
     * Constructor for the Matrix class.
     * @param data Pointer to the matrix data (column-major format).
     * @param width Width of the matrix.
     * @param ld Leading dimension of the matrix.
     * @param pivot Pointer to the pivot array.
     */
    __device__ Matrix(double* data, int width, int ld) : data(data), width(width), ld(ld){}

    /**
     * Accessor for elements of the matrix.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the matrix element at the specified row and column.
     */
    __device__ double& operator()(int row, int col) {
        return data[col * ld + row];
    }

    /**
     * Returns a Row object for the specified row index.
     * @param i The row index.
     * @return A Row object representing the specified row.
     */
    __device__ Row row(int i) {
        return Row(data + i, width, ld);
    }

    /**
     * Performs the work to bring the current row to row echelon form.
     * @param row Current row index.
     * @param col Current column index.
     * @param tolerance Tolerance for determining pivot validity.
     * @return True if a valid pivot was found; False otherwise.
     */
    __device__ bool rowEchelonWorkRow(int row, int col, double tolerance) {

	Row r(data + row, width, ld);
        int swapWith = row;

        // Find a valid pivot row
        while (fabs((*this)(swapWith, col)) <= tolerance && swapWith < width) 
            swapWith++;

	if(swapWith == width) return false;
	if(swapWith > row){
            Row needsSwap(data + swapWith, width, ld);
            r.swap(needsSwap);       
        }

        // Perform row elimination
        double diagonalElement = (*this)(row, col);
        for (int j = row + 1; j < width; j++) {
            double factor = (*this)(j, col) / diagonalElement;
            Row lower(data + j, width, ld);
            lower.subtract(r, factor);
        }
        return true;
    }

    /**
     * Converts the matrix to row echelon form via Gaussian elimination.
     * Updates the pivot index if a row swap occurs.
     * @param tolerance Tolerance for determining pivot validity.
     * @param trackPivots the indices of the pivot coluns will be set to one.
     */
    __device__ int rowEchelon(double tolerance) {
        int numPivots = 0, row = 0;
        for (int col = 0; col < width; col++)
            if (rowEchelonWorkRow(row, col, tolerance)) row++;
	    else numPivots++;	
	return numPivots;
    }

    /**
     * Prints the matrix in a readable row-major format.
     * Note: Use only for debugging purposes as printf is costly in CUDA kernels.
     */
    __device__ void printMatrix(const char* label = "") {
        if (threadIdx.x == 0 && blockIdx.x == 0) { // Ensure only one thread prints
            printf("Matrix %s (%dx%d):\n", label, width, width);
            for (int row = 0; row < width; row++) {
                for (int col = 0; col < width; col++) {
                    printf("%10.4f ", (*this)(row, col));
                }
                printf("\n");
            }
            printf("\n");
        }
    }  
};

/**
 * CUDA kernel to process a batch of matrices, converting each to row echelon form.
 * @param from Pointer to input matrix array (batch of column-major matrices).
 * @param ldFrom Leading dimension of input matrices.
 * @param to Pointer to output array for nullspace computation. The final element of each vector should be 1.
 * @param ldTo Leading dimension of output matrices.
 * @param batchSize Number of matrices in the batch.
 * @param width Width of each square matrix.
 * @param tolerance Tolerance for numerical operations.
 * @param workSpace Workspace for storing pivot information.
 */
extern "C" __global__ void nullSpace1dBatchKernel(int batchSize, double* from, int ldFrom, double* to, int ldTo, int width, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batchSize) return;

    Matrix mat(from + width * ldFrom * idx, width, ldFrom);

    double* eVec = to + ldTo * idx;
    
    int numPivots = mat.rowEchelon(tolerance);

    for(int i = width - 1; i > width - numPivots; i--) eVec[i] = 0;
    eVec[width - numPivots] = 1;
    
    for (int row = width - numPivots - 1, col = width - 1; row >= 0; row--,  col--) {
    	eVec[row] = 0;
        while(col > 0 && fabs(mat(row, col - 1)) > tolerance) col--;
        for (int i = col + 1; i <= width - numPivots; i++) eVec[row] -= eVec[i] * mat(row, i);
        eVec[row] /= mat(row, col);
    }

}
