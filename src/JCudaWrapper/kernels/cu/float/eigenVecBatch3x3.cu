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
    __device__ Get(const int inputIdx, const int height, const int downSampleFactorXY)
: idx(inputIdx * downSampleFactorXY), height(height), downSampleFactorXY(downSampleFactorXY) {}

    
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
		return downSampleFactorXY * (idx / height) * ld + idx % height;
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
__device__ void swap(double& a, double& b) {
    double temp = a;
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
    __device__ MaxAbs(int initArg, double initVal) : arg(initArg), val(initVal) {}

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
 * @class Matrix
 * @brief Represents a matrix and provides utility functions for matrix operations. * 
 */
class Matrix {
private:
    double mat[3][3];
    const double tolerance;
    

public:
    /**
     * @brief Constructor for Matrix.
     * @param xx, xy, xz, yy, yz, zz Matrix elements.
     * @param eVal Eigenvalue for computation.
     * @param tolerance Numerical tolerance for pivot detection.
     */
    __device__ Matrix(float xx, float xy, float xz, float yy, float yz, float zz, float eVal, float tolerance) 
    :tolerance(tolerance) {
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
    __device__ double& operator()(int row, int col) {
        return mat[row][col];
    }
    
    /**
     * @brief Subtracts a scaled row from another row.
     * @param minuendInd Index of the row to be updated.
     * @param subtrahendInd Index of the row to subtract.
     * @param scale Scaling factor.
     */
    __device__ void subtractRow(int minuendInd, int subtrahendInd, double scale) {
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
    __device__ bool reduceToRowEchelon(const int row, const int col) {
        
        MaxAbs maxPivot(row, fabs(mat[row][col]));
        
	for (int i = row + 1; i < 3; i++) maxPivot.challenge(i, mat[i][col]);

        if (maxPivot.getVal() <= tolerance*10) return false;

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
 * @class Vec
 * @brief A simple wrapper for a double array representing a 3D vector.
 */
class Vec {
private:
    float* data;
public:
    /**
     * @brief Constructs a Vec object.
     * @param data Pointer to the float array (size 3) representing the vector.
     */
    __device__ Vec(float* data):data(data){}

    /**
     * @brief Sets the components of the vector.
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     */
    __device__ void set(float x, float y, float z){
        data[0] = x; data[1] = y; data[2] = z;
    }

    /**
     * @brief Accesses a component of the vector using array-like indexing.
     * @param i The index of the component (0 for x, 1 for y, 2 for z).
     * @return A reference to the requested vector component.
     */
    __device__ float& operator[](int i) {
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
     * If the vector's length is zero, it remains unchanged.
     */
    __device__ void normalize() {
        double len = length();
        if (len > 0.0f) {
            double invLen = 1.0 / len;
            data[0] *= invLen;
            data[1] *= invLen;
            data[2] *= invLen;
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
 * @param eValues Pointer to all the eValues, including those that will not be used.  Be sure to increment valIndex over multiple runs of this kernel so that they are all used.Be sure the first values is the desired eigen value of each set of 3.
 * @param workspacePivotFlags Pointer to workspace memory for pivot flags.
 * @param tolerance Tolerance for row echelon pivot detection.
 * @param ldEVec the leading dimension of the eigen vectors.
 * @param ldSrc the leading dimension of the sourver matrix.
 * @param vecIndex The index of the desired eigen value. 
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
     
     const int downSampleFactorXY,
     int vecInd,
     float tolerance
) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize/downSampleFactorXY/downSampleFactorXY) return;
    
    Get getx3(3*idx, heightEVal, 1);
         
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
        tolerance
    );
    
    Vec vec(eVectors + getx3.ind(heightEVec, ldEVec));

    switch(mat.rowEchelon()){        
    
	case 1:
	    if(fabs(mat(0, 0)) <= tolerance) vec.set(1, 0, 0);
	    else if(fabs(mat(1, 1)) <= tolerance) vec.set(-mat(0,1)/mat(0,0), 1, 0);
	    else {
	        vec[2] = 1; 
	        vec[1] = -mat(1,2)/mat(1,1); 
	        vec[0] = (-mat(0,2) - mat(0,1)*vec[1])/mat(0,0);
	    }
	    break;
	case 2:
	    if(fabs(mat(0,0)) <= tolerance)
	        if(fabs(mat(0, 1)) <= tolerance)
	            if(vecInd % 2 == 0) vec.set(1, 0, 0);
	            else vec.set(0, 1, 0);
	        else if(vecInd % 2 == 0) vec.set(0, -mat(0, 2)/mat(0, 1), 1);
	            else vec.set(1, 0, 0);
	    else if(vecInd % 2 == 0) vec.set(-mat(0, 1)/mat(0, 0), 1, 0);
	    else vec.set(-mat(0, 2)/mat(0, 0), 0, 1);
	    break;
	case 3:
	    switch(vecInd){
	        case 0: vec.set(1, 0, 0); break;
	        case 1: vec.set(0, 1, 0); break;
	        case 2: vec.set(0, 0, 1);
	    }	    
    }
    
    if(idx == 1000*srcHeight + 580) {
        printf("vec in eigenVecBatch3x3 with index %d -> (%d, %d) is : (%f. %f, %f)\nwith tolerance %f\n", idx, idx/srcHeight, idx%srcHeight, vec[0], vec[1], vec[2], tolerance);
        mat.print();    
    }

    
//    if(idx == 0) {mat.print(); vec.print();}

    vec.normalize();
    
}
