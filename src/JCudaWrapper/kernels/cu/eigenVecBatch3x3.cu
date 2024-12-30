/**
 * @file eigenVecBatchKernel.cu
 * @brief CUDA kernel for computing eigenvectors using row echelon form.
 */

#include <cstdio>
#include <cmath>


#define DIM 3

/**
 * @brief Utility function to swap two double values.
 * @param a Reference to the first double.
 * @param b Reference to the second double.
 */
__device__ void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

/**
 * @class Row
 * @brief Represents a row of a matrix with utility functions for row operations.
 */
class Row {
private:
    double* data; ///< Pointer to the row data.

public:
    /**
     * @brief Constructor for Row.
     * @param data Pointer to the row data.
     */
    __device__ Row(double* data) : data(data) {}

    /**
     * @brief Access an element in the row by column index.
     * @param col Column index.
     * @return Reference to the element at the specified column.
     */
    __device__ double& operator()(int col) {
        return data[col * DIM];
    }

    /**
     * @brief Subtracts a scaled version of another row from this row.
     * @param other The row to subtract.
     * @param timesOther Scaling factor for the other row.
     */
    __device__ void subtract(Row& other, double timesOther) {
        for (int i = 0; i < DIM; i++) 
            (*this)(i) -= timesOther * other(i);
    }

    /**
     * @brief Swaps this row with another row.
     * @param other The row to swap with.
     */
    __device__ void swap(Row& other) {
        for (int i = 0; i < DIM; i++) 
            ::swap((*this)(i), other(i));
    }
};

/**
 * @class Matrix
 * @brief Represents a matrix and provides utility functions for matrix operations.
 */
class Matrix {
private:
    double* data; ///< Pointer to the matrix data.

public:
    /**
     * @brief Constructor for Matrix.
     * @param data Pointer to the matrix data.
     */
    __device__ Matrix(double* data) : data(data){}

    /**
     * @brief Access an element in the matrix by row and column index.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at the specified row and column.
     */
    __device__ double& operator()(int row, int col) {
        return data[col * DIM + row];
    }

    /**
     * @brief Get a Row object for the specified row index.
     * @param i Row index.
     * @return Row object for the specified row.
     */
    __device__ Row row(int i) {
        return Row(data + i);
    }

    /**
     * @brief Perform row echelon work for a specific row and column.
     * @param row Row index.
     * @param col Column index.
     * @param tolerance Tolerance for considering a pivot.
     * @return True if a pivot was found, false otherwise.
     */
    __device__ bool rowEchelonWorkRow(int row, int col, double tolerance) {
        Row r(data + row);

        double maxPivot = fabs((*this)(row, col));
        int maxRow = row;

        for (int i = row + 1; i < DIM; i++) {
            double absVal = fabs((*this)(i, col));
            if (absVal > maxPivot) {
                maxPivot = absVal;
                maxRow = i;
            }
        }

        if (maxRow == -1 || maxPivot <= tolerance) return false;

        if (maxRow != row) {
            Row needsSwap(data + maxRow);
            r.swap(needsSwap);
        }

        double diagonalElement = (*this)(row, col);
        for (int j = row + 1; j < DIM; j++) {
            double factor = (*this)(j, col) / diagonalElement;
            Row lower(data + j);
            lower.subtract(r, factor);
        }
        return true;
    }

    /**
     * @brief Perform row echelon reduction on the matrix.
     * @param tolerance Tolerance for considering a pivot.
     * @return Number of pivots found during the reduction.
     */
    __device__ void rowEchelon(double tolerance) {
        
        for (int col = 0, row = 0; col < DIM; col++)
            if (rowEchelonWorkRow(row, col, tolerance)) row++;
    }

    /**
     * @brief Print the matrix to the console for debugging.
     * @param label Optional label to display with the matrix.
     */
    __device__ void printMatrix(const char* label = "") {
        printf("\n");
        
        printf("Thread id: %d\n", blockIdx.x * blockDim.x + threadIdx.x);
        
        for (int row = 0; row < DIM; row++) {
            for (int col = 0; col < DIM; col++) printf("%10.4f ", (*this)(row, col));
	    printf("\n");
        }
        printf("\n");
    }
    
};

/**
 * @brief Sets the matrix to A - \lambda I.
 * @param from Pointer to the source matrix.
 * @param to Reference to the destination Matrix object.
 * @param eVal Eigenvalue \lambda.
 */
__device__ void setAMinusLambdaI(const double* from, Matrix& to, const double& eVal) {

    for (int i = 0; i < DIM * DIM; i++) to(i / DIM, i % DIM) = from[i];
    for (int i = 0; i < DIM; i++) to(i, i) -= eVal;
}

class Vector{
private:
    double* data;
public:
    __device__ Vector(double* data): data(data){}
    
    /**
     * @brief Set vector components.
     * @param a Component along x-axis.
     * @param b Component along y-axis.
     * @param c Component along z-axis.
     */
    __device__ void set(double a, double b, double c){
        data[0] = a;
        data[1] = b;
        data[2] = c;
    }
    __device__ double& operator[](int i) {
        return data[i];
    }
};

/**
 * @brief CUDA kernel to compute eigenvectors in batch using row echelon form.
 * @param batchSize Number of matrices in the batch.
 * @param sourceMatrices Pointer to source matrices in column-major format.
 * @param ldSourceMatrices Leading dimension of the source matrices.
 * @param eVectors Pointer to the resulting eigenvectors.
 * @param ldEVecs Leading dimension of the eigenvector array.
 * @param eVals Pointer to the eigenvalues.
 * @param workSpace Pointer to workspace memory for intermediate computations.
 * @param tolerance Tolerance for row echelon pivot detection.
 */
extern "C" __global__ void eigenVecBatch3x3Kernel(
    const int batchSize, 
    const double* sourceMatrices, 
    const int ldSourceMatrices, 
    double* eVectors, 
    const int ldEVecs,
    const double* eVals,
    double* workSpace,
    const double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batchSize) return;
    
    Matrix m(workSpace + idx * DIM*DIM);
    setAMinusLambdaI(sourceMatrices + idx * DIM*DIM, m, eVals[idx]);

    Vector v(eVectors + ldEVecs * idx);

    m.rowEchelon(tolerance);

    if(idx == 1) m.printMatrix();

    if(fabs(m(0,0)) > tolerance){
        
        if(fabs(m(1,1)) > tolerance){
    	    v[2] = 1; 
    	    v[1] = -m(1,2)/m(1,1); 
    	    v[0] = -(m(0,1)*v[1] - m(0,2))/m(0,0);
        }
        else {
            if(fabs(m(1,2))  > tolerance) v.set(-m(0,1)/m(
            0,0), 1, 0);
            else {
                if(idx % 3 == 0) v.set(-m(0,2)/m(0,0), 0, 1);
                else if(idx % 2 == 0) v.set(-m(0,1)/m(0,0), 1, 0);
                else v.set(-(m(0,1) + m(0,2))/m(0,0), 1, 1);
            }
        }
    }else{
    	if(fabs(m(0,1)) > tolerance){
    		if(fabs(m(1,2)) > tolerance) v.set(1,0,0);
    		else{
    		    if(idx % 2 == 0) v.set(1, 0, 0);
    		    else v.set(0, -m(0,2)/m(0,1), 1);//maybe has multiplicity of 3
    		}
    	}
    	else{
    	    if(idx%2 == 0) v.set(1,0,0);
    	    else v.set(0,1,0);
    	}
    }
}

