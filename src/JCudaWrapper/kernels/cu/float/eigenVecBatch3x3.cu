/**
 * @file eigenVecBatchKernel.cu
 * @brief CUDA kernel for computing eigenvectors using row echelon form.
 */

#include <cstdio>
#include <cmath>

/**
 * @class Get
 * @brief A helper class for accessing values in a column-major order matrix.
 */
class Get{
private:
    const int height;
    const int idx;
    const int downSampleFactorXY;
public:
    __device__ Get(const int idx, const int height, const int downSampleFactorXY): idx(idx), height(height), downSampleFactorXY(downSampleFactorXY){}
    
    /**
     * @brief Retrieves a value from a column-major order matrix.
     * @param src Pointer to the source array.
     * @param ld The leading dimension (stride between columns in memory).
     * @return The value at the corresponding column-major index.
     */
    __device__ float val(const float* src, const int  ld) const{
		return val(src, height, ld);		
    }
    
    /**
     * @brief Retrieves a value from a column-major order matrix.
     * @param src Pointer to the source array.
     * @param ld The leading dimension (stride between columns in memory).
     * @param height The height of the matrix.
     * @return The value at the corresponding column-major index.
     */
    __device__ float val(const float* src, const int height, const int ld) const{
		return src[ind(height, ld)];
    }
    
    /**
     * @brief Retrieves an index from a column-major order matrix.
     * @param height The height of the matrix.
     * @param ld The leading dimension (stride between columns in memory).
     * @return The computed column-major index.
     */
    __device__ int ind(const int height, const int ld) const{
		return (downSampleFactorXY * (idx / height)) * ld + downSampleFactorXY * (idx % (height / downSampleFactorXY));
    }
    
    /**
     * @brief Retrieves an index from a column-major order matrix using stored height.
     * @param ld The leading dimension (stride between columns in memory).
     * @return The computed column-major index.
     */
    __device__ int ind(const int ld) const{
		return ind(height, ld);
    }
    
    
};


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
 * @brief Represents a matrix and provides utility functions for matrix operations. * 
 */
class Matrix {
private:
    float mat[3][3];     
    int* isPivot; ///< Pointer to an array indicating pivot columns.
    const float tolerance;
    

public:
    /**
     * @brief Constructor for Matrix.
     * @param xx, xy, xz, yy, yz, zz Matrix elements.
     * @param eVal Eigenvalue for computation.
     * @param isPivot Pointer to pivot flag array.
     * @param tolerance Numerical tolerance for pivot detection.
     */
    __device__ Matrix(float xx, float xy, float xz, float yy, float yz, float zz, float eVal, int* isPivot, float tolerance) 
    : isPivot(isPivot), tolerance(tolerance) {
        mat[0][0] = xx - eVal; mat[0][1] = xy; mat[0][2] = xz;
        mat[1][0] = xy; mat[1][1] = yy - eVal; mat[1][2] = yz;
        mat[2][0] = xz; mat[2][1] = yz; mat[2][2] = zz - eVal;
    }


    /**
     * @brief Access an element in the matrix by row and column index.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at the specified row and column.
     */
    __device__ float& operator()(int row, int col) {
        return mat[row][col];
    }
    
    /**
     * @brief Subtracts a scaled row from another row.
     * @param minuendInd Index of the row to be updated.
     * @param subtrahendInd Index of the row to subtract.
     * @param scale Scaling factor.
     */
    __device__ void subtractRow(int minuendInd, int subtrahendInd, float scale) {
        for (int i = 0; i < 3; i++) mat[minuendInd][i] -= scale * mat[subtrahendInd][i];        
    }

    /**
     * @brief Swaps two rows of the matrix.
     * @param i First row index.
     * @param j Second row index.
     */
    __device__ void swapRows(int i, int j) {
        for(int k = 0; k < 3; k++) swap(mat[i][k], mat[j][k]);
    }

    
    /**
     * @brief Perform row echelon work for a specific row and column.
     * @param row Current row index.
     * @param col Current column index.
     * @return True if a pivot was found, false otherwise.
     */
    __device__ bool reduceToRowEchelon(int row, int col) {
        
        MaxAbs maxPivot(row, fabs(mat[row][col]));
        
	for (int i = row + 1; i < 3; i++) maxPivot.challenge(i, mat[i][col]);

        if (maxPivot.getVal() <= tolerance) return false;

        if (maxPivot.getArg() != row) swapRows(maxPivot.getArg(), row);
        
        for (int i = row + 1; i < 3; i++)
	    subtractRow(i, row, mat[i][col]/mat[row][col]);

        return true;
    }

    /**
     * @brief Perform row echelon reduction on the matrix.
     * @return Number of free variables found during the reduction.
     */
    __device__ int rowEchelon() {
        int numFreeVariables = 0;
        int row = 0;

        for (int col = 0; col < 3; col++) {            
            if (reduceToRowEchelon(row, col)) {
                row++;
                isPivot[col] = 1;
            } else {
                isPivot[col] = 0;
                numFreeVariables++;
            }
        }

        if (fabs(mat[0][0]) < tolerance) isPivot[0] = 0;

        return numFreeVariables;
    }
    
    /**
     * Prints the matrix for debuging purposes.
     */
    __device__ void print() {
        printf("Matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%f ", mat[i][j]);
            }
            printf("\n");
        }
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
 * @param eValues Pointer to all the eValues, including those that will not be used.  Be sure to increment valIndex over multiple runs of this kernel so that they are all used.
 * @param workspacePivotFlags Pointer to workspace memory for pivot flags.
 * @param tolerance Tolerance for row echelon pivot detection.
 * @param ldEVec the leading dimension of the eigen vectors.
 * @param ldSrc the leading dimension of the sourver matrix.
 * @param valIndex The index of the desired eigen value. 
 */
extern "C" __global__ void eigenVecBatch3x3Kernel(
     const int batchSize, 
     const float* xx, const int ldxx, 
     const float* xy, const int ldxy, 
     const float* xz, const int ldxz,
     const float* yy, const int ldyy,
     const float* yz, const int ldyz,
     const float* zz, const int ldzz, 
     const int srcHeight, 
    
     float* eVectors,
     const int ldEVec,
     const int heightEVec,
         
     const float* eValues,
     const int ldEVal,
     const int heightEVal,
     
     int* workspacePivotFlags,
     const int ldPivot,
     const int heightPivot, 
          
     const float tolerance,
     const int downSampleFactorXY
) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize/downSampleFactorXY/downSampleFactorXY) return;
    
    Get getx3(3*idx, heightEVal, 1);
    
    int* isPivot = workspacePivotFlags + getx3.ind(heightPivot, ldPivot);    
    
    float eVal = getx3.val(eValues, ldEVal);    
    
    Get get(idx, srcHeight, downSampleFactorXY);
    Matrix mat(
        get.val(xx, ldxx), 
        get.val(xy, ldxy), 
        get.val(xz, ldxz), 
        get.val(yy, ldyy), 
        get.val(yz, ldyz), 
        get.val(zz, ldzz),
        eVal,
        isPivot,
        tolerance
    );
    
    if(idx == 1) {
	mat.print();
	printf("eVal = %f\n", eVal);
    }
    
    float* eVec = eVectors + getx3.ind(heightEVec, ldEVec);
    int numFreeVariables = mat.rowEchelon();

    int col = 2;
    
    while(isPivot[col]) eVec[col--] = 0;    

    eVec[col--] = 1;
    
    for (int row = col; row >= 0 && col >= 0; col--) {	
    	eVec[col] = 0;	
		if(isPivot[col]){         
            for (int i = col + 1; i < 3; i++) 
                eVec[col] -= eVec[i] * mat(row, i);
            eVec[col] /= mat(row, col);
            row--;
        }
    }
 //   if(idx == 0) printf("eVec = (%f, %f, %f)\n", eVec[0], eVec[1], eVec[2]);//mat.print();
}
