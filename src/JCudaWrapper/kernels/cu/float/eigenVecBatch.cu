/**
 * @file eigenVecBatchKernel.cu
 * @brief CUDA kernel for computing eigenvectors using row echelon form.
 */

#include <cstdio>
#include <cmath>

/**
 * @brief Utility function to swap two float values.
 * @param a Reference to the first float.
 * @param b Reference to the second float.
 */
__device__ void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

/**
 * @class MatrixRow
 * @brief Represents a single row of a matrix with utility functions for row operations.
 */
class MatrixRow {
private:
    float* elements; ///< Pointer to the row data.
    const int width;  ///< Number of columns in the matrix.
    const int ld;   ///< Leading dimension of the matrix.

public:
    /**
     * @brief Constructor for MatrixRow.
     * @param elements Pointer to the row data.
     * @param width Number of columns in the matrix.
     * @param height Leading dimension of the matrix.
     */
    __device__ MatrixRow(float* elements, int width, int ld) 
        : elements(elements), width(width), ld(ld) {}

    /**
     * @brief Access an element in the row by column index.
     * @param col Column index.
     * @return Reference to the element at the specified column.
     */
    __device__ float& operator()(int col) {
        return elements[col * ld];
    }

    /**
     * @brief Subtracts a scaled version of another row from this row.
     * @param other The row to subtract.
     * @param scale Scaling factor for the other row.
     */
    __device__ void subtract( MatrixRow& other, float scale) {
        for (int i = 0; i < width; i++) (*this)(i) -= scale * other(i);        
    }

    /**
     * @brief Swaps this row with another row.
     * @param other The row to swap with.
     */
    __device__ void swap(MatrixRow& other) {
        for (int i = 0; i < width; i++) ::swap((*this)(i), other(i));
    }
};

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
    float val; ///< The maximum absolute value encountered so far.    

public:
    /**
     * @brief Constructor for the MaxAbs class.
     * 
     * Initializes the maximum absolute value and its corresponding argument.
     *
     * @param initVal The initial maximum absolute value.
     * @param initArg The initial argument corresponding to the maximum absolute value.
     */
    __device__ MaxAbs(int initArg, float initVal) : arg(initArg), val(initVal) {}

     /**
     * @brief Updates the tracked maximum absolute value if the new value is greater.
     * 
     * Compares the given value with the current maximum absolute value. If the new value is greater,
     * updates the maximum value and its corresponding index.
     *
     * @param candidateIndex The index associated with the new value.
     * @param candidateValue The new value to compare against the current maximum absolute value.
     */
    __device__ void challenge(int candidateIndex, float candidateValue) {
        float absoluteValue = fabs(candidateValue); // Compute the absolute value of the candidate value.
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
    __device__ float getVal() {
        return val;
    }
};


/**
 * @class Matrix
 * @brief Represents a matrix and provides utility functions for matrix operations.
 */
class Matrix {
private:
    float* data; ///< Pointer to the matrix data.    
    int* isPivot; ///< Pointer to an array indicating pivot columns.
    const float tolerance;
    

public:
     const int width; ///< Number of columns in the matrix.
     const int height; ///< Number of rows in the matrix.
     const int ld;
    /**
     * @brief Constructor for Matrix.
     * @param data Pointer to the matrix data.
     * @param width Number of columns in the matrix.
     * @param height Number of rows in the matrix.
     * @param isPivot Pointer to an array indicating pivot columns.
     */
    __device__ Matrix(float* data, int width, int height, int ld, int* isPivot, float tolerance) 
        : data(data), width(width), height(height), isPivot(isPivot), tolerance(tolerance), ld(ld) {}

    /**
     * @brief Access an element in the matrix by row and column index.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at the specified row and column.
     */
    __device__ float& operator()(int row, int col) {
        return data[col * ld + row];
    }


    /**
     * @brief Get a MatrixRow object for the specified row index.
     * @param rowIndex Row index.
     * @return MatrixRow object for the specified row.
     */
    __device__ MatrixRow getRow(int rowIndex) {
        return MatrixRow(data + rowIndex, width, ld);
    }
    
    /**
     * @brief Perform row echelon work for a specific row and column.
     * @param rowInd Current row index.
     * @param col Current column index.
     * @return True if a pivot was found, false otherwise.
     */
    __device__ bool reduceToRowEchelon(int rowInd, int col) {
        
        MatrixRow row = getRow(rowInd);
        
        MaxAbs maxPivot(rowInd, fabs((*this)(rowInd, col)));
	for (int i = rowInd + 1; i < height; i++) maxPivot.challenge(i, (*this)(i, col));

        if (maxPivot.getVal() <= tolerance) return false;

        if (maxPivot.getArg() != rowInd) getRow(maxPivot.getArg()).swap(row);        
        
        for (int i = rowInd + 1; i < height; i++)
	    getRow(i).subtract(row, (*this)(i, col)/(*this)(rowInd, col));

        return true;
    }

    /**
     * @brief Perform row echelon reduction on the matrix.
     * @return Number of free variables found during the reduction.
     */
    __device__ int rowEchelon() {
        int numFreeVariables = 0;
        int row = 0;

        for (int col = 0; col < width; col++) {            
            if (reduceToRowEchelon(row, col)) {
                row++;
                isPivot[col] = 1;
            } else {
                isPivot[col] = 0;
                numFreeVariables++;
            }
        }

        if (fabs((*this)(0, 0)) < tolerance) isPivot[0] = 0;

        return numFreeVariables;
    }

    /**
     * @brief Print the matrix to the console for debugging.
     * @param label Optional label to display with the matrix.
     */
    __device__ void print( char* label = "") {
        
            printf("Matrix %s (%dx%d):\n", label, width, height);
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    printf("%10.4f ", (*this)(row, col));
                }
                printf("\n");
            }
            printf("\n");
        }
    
};


/**
 * This method should be called on a fresh copy of the matrices for which the vectors are sought for each eigenvalue.  Each time with an incremented value of valIndex.
 *
 * @brief CUDA kernel to compute eigenvectors in batch using row echelon form.
 * @param batchSize Number of matrices.
 * @param src Pointer to source matrices in column-major format.  These matrices will be changed.
 * @param ldsrc Leading dimension of the source matrices.
 * @param eVectors Pointer to the resulting eigenvectors.
 * @param width Number of columns in each matrix.
 * @param height Number of rows in each matrix.
 * @param eigenValues Pointer to all the eigenValues, including those that will not be used.  Be sure to increment valIndex over multiple runs of this kernel so that they are all used.
 * @param workspacePivotFlags Pointer to workspace memory for pivot flags.
 * @param tolerance Tolerance for row echelon pivot detection.
 * @param ldEVec the leading dimension of the eigan vectors.
 * @param ldSrc the leading dimension of the sourver matrix.
 * @param valIndex The index of the desired eigan value. 
 */
 //TODO: add option of only finding the first eigenVector.  
 //TODO: can this be accelerated for 3x3 as opposed to generic?
extern "C" __global__ void eigenVecBatchKernel(
     const int batchSize, 
     float* src, 
     const int height,
     const int width,
     const int ldSrc,
     float* eVectors,
     const int ldEVec,
     const int valIndex,     
     const float* eigenValues,
     const int ldEVal,
     int* workspacePivotFlags,
     const float tolerance
) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    // Set up the pivot flags for the current matrix.
    int* isPivot = workspacePivotFlags + width * idx;
    
    Matrix mat(src + idx * ldSrc * width, width, height, ldSrc, isPivot, tolerance);
    
    float eiganVal = eigenValues[idx * ldEVal + valIndex];
    for(int i = 0; i < width; i++) mat(i, i) -= eiganVal;

    float* eVec = eVectors + ldEVec * width * idx + valIndex * ldEVec;
    int numFreeVariables = mat.rowEchelon();
    
    //printf("%d free variables -> %d\n",idx, numFreeVariables);//mat.print();
    
    int freeVariableID = valIndex % numFreeVariables; 

    int col = width - 1, pivotsPassed = 0;
    // Loop through the columns in reverse, starting from the last column.
    // All the values of the eigenvector after the free variable are set to 0.  
    // The value of the eigenvector that corresponds to the free vector is set to 1.
    for (int idfv = freeVariableID; idfv >= 0; col--) {
		if (!isPivot[col]) idfv--;
		else pivotsPassed++;
		eVec[col] = 0;
    }
    
    if(col < width - 1) col++;
    eVec[col] = 1;
    col--;
    
    for (int row = (width - numFreeVariables) - pivotsPassed - 1; row >= 0 && col >= 0; col--) {	
    	eVec[col] = 0;	
		if(isPivot[col]){         
            for (int i = col + 1; i < width - freeVariableID; i++) 
                eVec[col] -= eVec[i] * mat(row, i);
            eVec[col] /= mat(row, col);
            row--;
        }
    }    
}
