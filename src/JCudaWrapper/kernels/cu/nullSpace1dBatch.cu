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
    int width;
    int ld;

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
     * Computes the pointer to an element in the row based on the column index.
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
    int* pivot;
    int width; /**< Width of the matrix. */
    int ld;

public:
    /**
     * Constructor for the Matrix class.
     * @param data Pointer to the matrix data (column-major format).
     * @param width Width of the matrix.
     * @param ld Leading dimension of the matrix.
     * @param pivot Pointer to the pivot array.
     */
    __device__ Matrix(double* data, int width, int ld, int* pivot) : data(data), width(width), ld(ld), pivot(pivot) {}

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
     * Converts the matrix to upper triangular form via Gaussian elimination.
     * Updates the pivot index if a row swap occurs.
     * @param tolerance Tolerance for determining pivot validity.
     */
    __device__ void triangulize(double tolerance) {
        for (int i = 0; i < width; i++) {

            Row r(data + i, width, ld);

            int swapWith = i;
            
            while (fabs((*this)(swapWith, i)) <= tolerance && swapWith < width) 
                swapWith++;
            if (swapWith != i && swapWith < width) {
                Row needsSwap(data + swapWith, width, ld);
                r.swap(needsSwap);
            }
            
            pivot[i] = swapWith;
            
            double diagonalLmnt = (*this)(i, i);

            for (int j = i + 1; j < width; j++) {
                double factor = (*this)(j, i) / diagonalLmnt;
                Row lower(data + j, width, ld);
                lower.subtract(r, factor);
            }
        }
    }
    
    __device__ void reversePivot(double* vec){
    	for(int i = width - 1; i >= 0; i--){
    	    if(pivot[i] != i){
    	    	Row r1(data + i, width, ld);
    	    	Row r2(data + pivot[i], width, ld);
    	    	r1.swap(r2);
    	    }
    	}
    		
    }
};

/**
 * CUDA kernel to process a batch of matrices, converting each to upper triangular form.
 * @param from Pointer to input matrix array (batch of column-major matrices).
 * @param ldFrom Leading dimension of input matrices.
 * @param to Pointer to output array for nullspace computation.  The final element of each vector should be 1.
 * @param ldTo Leading dimension of output matrices.
 * @param batchSize Number of matrices in the batch.
 * @param width Width of each square matrix.
 * @param tolerance Tolerance for numerical operations.
 * @param workSpace Workspace for storing pivot information.
 */
extern "C" __global__ void nullSpace1dBatchKernel(double* from, int ldFrom, double* to, int ldTo, int batchSize, int width, double tolerance, int* workSpace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batchSize) return;

    Matrix mat(from + width * ldFrom * idx, width, ldFrom, workSpace + width * idx);
    mat.triangulize(tolerance);

    double* eVec = to + ldTo * idx;

    for(int i = width - 2; i >= 0; i--){
    	eVec[i] = 0;
    	for(int j = i + 1; j < width; j++) eVec[i] -= eVec[j] * mat(i, j);
    	eVec[i] /= mat(i, i); 
    }

    mat.reversePivot(eVec);
}
