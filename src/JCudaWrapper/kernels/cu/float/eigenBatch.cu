#include <cuda_runtime.h>
#include <math.h>


/**
 * Uses Kahan's method for more accurate mulitplication.
 */
__device__ double prod(double a, double b){
    double result = a*b;
    return result - fma(a, b, -result);
}

/**
 * Swap function for double values.
 *
 * @param a First value.
 * @param b Second value.
 */
__device__ inline void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}
/**
 * Swap function for double values.
 *
 * @param a First value.
 * @param b Second value.
 */
__device__ inline void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

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
    __device__ MaxAbs(int initArg, double initVal) : arg(initArg), val(fabs(initVal)) {}

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
 * Represents a 3x3 symmetric matrix in column-major format.
 */
class Matrix3x3 {
private:
    double mat[3][3];
    double tolerance;
    
    /**
     * A value that is less than tolerance will be returned as 0.  Otherwise as itself.
     */
    __device__ double zeroBar(double maybeNear0){
        return fabs(maybeNear0) <= tolerance? 0: maybeNear0;
    }
public:
    /**
     * Constructs a Matrix3x3 object.
     *
     * @param xx Element at (0,0).
     * @param xy Element at (0,1) and (1,0).
     * @param xz Element at (0,2) and (2,0).
     * @param yy Element at (1,1).
     * @param yz Element at (1,2) and (2,1).
     * @param zz Element at (2,2).
     * @param tol the tolerance.
     */
    __device__ explicit Matrix3x3(const float xx, const float xy, const float xz, const float yy, const float yz, const float zz, double tol) : tolerance(tol) {
        mat[0][0] = zeroBar(xx);
        mat[0][1] = mat[1][0] = zeroBar(xy);
        mat[0][2] = mat[2][0] = zeroBar(xz);
        mat[1][1] = zeroBar(yy);
        mat[1][2] = mat[2][1] = zeroBar(yz);
        mat[2][2] = zeroBar(zz);
    }



    /**
     * Computes the trace of the matrix.
     * @return The sum of the diagonal elements.
     */
    __device__ double trace() const {
        return mat[0][0] + mat[1][1] + mat[2][2];
    }

    /**
     * Computes the sum of 2x2 determinant minors of the matrix.
     * @return The sum of determinant minors.
     */
    __device__ double diagMinorSum() const {
        return mat[1][1]*mat[2][2] - mat[1][2]*mat[1][2] + mat[0][0]*mat[2][2] - mat[0][2]*mat[0][2] + mat[0][0]*mat[1][1] - mat[0][1]*mat[0][1];
    }

    /**
     * Computes the determinant of the matrix.
     * @return The determinant value.
     */
    __device__ double determinant() const {
        return mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[1][2]) -
               mat[0][1] * (mat[0][1] * mat[2][2] - mat[1][2] * mat[0][2]) +
               mat[0][2] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
    }
    
    /**
     * Subtracts the val from each element on the diagnal of this matrix, changing this matrix.
     * @param val The value to be subtracted from each element of this matrix.
     */
    __device__ void subtractFromDiag(double val){
        mat[0][0] -= val; mat[1][1] -= val; mat[2][2] -= val;
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
     * @param The value at this column will be set to 0.  Perform subtraction to the right.
     */
    __device__ void subtractRow(int minuendInd, int subtrahendInd, double scale, int startCol) {
        mat[minuendInd][startCol] = 0;
        for (int i = startCol + 1; i < 3; i++){
            if(fabs(mat[minuendInd][i]) <= tolerance && fabs(mat[subtrahendInd][i]) <= tolerance) mat[minuendInd][i] = 0;
            else if(fabs(mat[minuendInd][i]) <= tolerance) mat[minuendInd][i] = prod(scale, mat[subtrahendInd][i]);
            else if(fabs(mat[subtrahendInd][i]) <= tolerance) mat[minuendInd][i] = -mat[minuendInd][i];
            else 
                mat[minuendInd][i] = fma(scale, mat[subtrahendInd][i], -mat[minuendInd][i]);
        }
        
    }

    /**
     * @brief Swaps two rows of the matrix.
     * @param i First row index.
     * @param j Second row index.
     * @param startCol begin swaping with this column and proceed to the right.
     */
    __device__ void swapRows(int i, int j, int startCol) {
        for(int k = startCol; k < 3; k++) swap(mat[i][k], mat[j][k]);
    }
    
    /**
     * Scales the row so that the element at the startCol is one and every element after is times one over that element.
     * @param row the row to be scaled.
     * @startCol the column index of the first non zero element of the row.
     */
    __device__ void scaleRow(int row, int startCol){
	
	double inv = 1/mat[row][startCol]; 
    	mat[row][startCol] = 1;
    	for(int i = startCol + 1; i < 3; i++) mat[row][i] *= inv;
    	
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
	

        if (maxPivot.getVal() <= tolerance) return false;

        if (maxPivot.getArg() != row) swapRows(maxPivot.getArg(), row, col);
        
        for (int i = row + 1; i < 3; i++){
	    subtractRow(i, row, mat[i][col]/mat[row][col], col);
	    scaleRow(row, col);
	}

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
 * Represents an affine function y = ax + b.
 */
class Affine {
private:
    double a; /**< The slope of the line. */
    double b; /**< The y-intercept of the line. */

public:
    /**
     * Constructs an Affine function.
     * @param a The slope.
     * @param b The y-intercept.
     */
    __device__ Affine(double a, double b) : a(a), b(b) {}

    /**
     * Evaluates the function at a given x.
     * @param x The input value.
     * @return The corresponding y-value.
     */
    __device__ double operator()(double x) {
        return fma(a, x, b);
    }

    /**
     * Maps multiple x-values to y-values.
     * @param x1 First x-value.
     * @param x2 Second and thid x-values.
     * @param y Pointer to an array where results are stored.
     */
    __device__ void map(double x1, double x2And3, double* y) {
        y[0] = (*this)(x1);
        y[1] = y[2] = (*this)(x2And3);
    }
    
    /**
     * @return The slope of the function.
     */
    __device__ double getSlope(){
        return a;
    }
    
    /**
     * Prints the function parameters.
     */
    __device__ void print(){
        printf("a = %lf and b = %lf\n\n", a, b);
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

class EVal{
private:
    double data[3];
    
    /**
     * Sorts an array in descending order.
     */
    __device__ void sortDescending() {
        if(data[0] < data[1]) swap(data[0], data[1]);
        if(data[0] < data[2]) swap(data[0], data[2]);
        if(data[1] < data[2]) swap(data[1], data[2]);
    }
    
    
    /**
     * @brief Sets the components of the vector.
     * @param x The x-component.
     * @param y The y-component.
     * @param z The z-component.
     */
    __device__ void set(double x, double y, double z){
        data[0] = x; data[1] = y; data[2] = z;
    }
    
    /**
     * Writes these values to the desired location.
     */
    __device__ void writeTo(float* to){
        to[0] = data[0]; to[1] = data[1]; to[2] = data[2];
    }

    /**
     * Computes the real roots of a cubic equation.
     *
     * @param b Coefficient of x^2.
     * @param c Coefficient of x.
     * @param d Constant term.
     * @param eigenInd The index of the eigenvalue to be returned from this method.  0 for the largest eigenValue and 2 for the smallest.
     * @param val Output array to store roots.
     * @return The eigen value at the desired index.
     */
    __device__ void cubicRoot(const double b, const double c, const double d){
	
	double inv3 = 1.0/3;
	
	double nBInv3 = -b*inv3;
	
	double p = fma(nBInv3, b, c) * inv3;
	double q = fma(fma(b/13.5, b, -c*inv3), b, d);

	if (p >= -1e-9) set(nBInv3, nBInv3, nBInv3);
	
	else{
	    
	    Affine line(2 * sqrt(-p), nBInv3);
	
	    double arg = q/prod(line.getSlope(), p);
	
	    if(arg > 1 - 1e-6) line.map(1, -0.5, data);
	    else if(arg < -1 + 1e-6) line.map(-1, 0.5, data);
	    else {

	        double acosArg = acos(arg); 

	        set(line(cos(acosArg * inv3)), 
 	            line(cos(fma(2, M_PI, acosArg) * inv3)), 
	            line(cos(fma(4, M_PI, acosArg) * inv3))
	    	);
	    }
	    		   
	   if(blockIdx.x * blockDim.x + threadIdx.x == 575*1153 + 150){
	       printf("eigenBatch Has eigenvalues (%lf, %lf, %lf)\n", data[0], data[1], data[2]);
	   }		   
	}
    }
public:

    /**
     * Finds the eigenvalues.
     *@param mat The matrix for whom the eigenvalues are desired.
     */
    __device__ EVal(const Matrix3x3& mat, float* dst){
       
 
        cubicRoot(-mat.trace(), mat.diagMinorSum(), -mat.determinant());
        sortDescending();
        writeTo(dst);
    }
    
    __device__ int multiplicity(int ind){
        return (data[0] == data[ind]) + (data[1] == data[ind]) + (data[2] == data[ind]) - 1;
    }
    
    
    /**
     * @brief Accesses a component of the vector using array-like indexing.
     * @param i The index of the component (0 for x, 1 for y, 2 for z).
     * @return A reference to the requested vector component.
     */
    __device__ double& operator[](int i) {
        return data[i];
    }
};


/**
 * CUDA Kernel to compute eigenvalues of a batch of 3x3 symmetric matrices.
 *
 * @param n Number of matrices fordownSampleFactorXY = 1, even if it's not.
 * @param srcHeight Height of the input matrices.
 * @param valDst Pointer to the output eigenvalues.
 * @param ldEVal Leading dimension of output.
 * @param 1 of every how many structure tensors should be evaluated in the x and y dimensions.
 */
extern "C" __global__ void eigenBatchKernel(
    const int n, 
    
    const float* xx, const int ldxx, 
    const float* xy, const int ldxy, 
    const float* xz, const int ldxz,
    const float* yy, const int ldyy,
    const float* yz, const int ldyz,
    const float* zz, const int ldzz, 
    
    const int srcHeight, 
    
    float* valDst, const int ldEVal, int heightValDst, 
    
    const int downSampleFactorXY, const int eigenInd,
    
    float* vecDst, const int ldEVec, const int heightEVec,
    
    double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n/downSampleFactorXY/downSampleFactorXY) return;
    
    Get src(idx, srcHeight, downSampleFactorXY);
    
    Matrix3x3 mat(
    	src.val(xx, ldxx), src.val(xy, ldxy), src.val(xz, ldxz), 
                           src.val(yy, ldyy), src.val(yz, ldyz), 
    					      src.val(zz, ldzz),
        tolerance
    );
    
    Get getx3(3*idx, heightEVec, 1);
    
    EVal eVals(mat, valDst + getx3.ind(ldEVal));
    
    mat.subtractFromDiag(eVals[eigenInd]);
    
    if(idx == 575*srcHeight + 150) mat.print();
    
    Vec vec(vecDst + getx3.ind(ldEVec));
    
    int freeVariables = mat.rowEchelon();
    
    double smTol = 1e-8;
    
    switch(freeVariables){
    
	case 1:
	    if(fabs(mat(0, 0)) <= smTol) vec.set(1, 0, 0);
	    else if(fabs(mat(1, 1)) <= smTol) vec.set(-mat(0,1)/mat(0,0), 1, 0);
	    else {
	        vec[2] = 1; 
	        vec[1] = -mat(1,2)/mat(1,1); 
	        vec[0] = (-mat(0,2) - mat(0,1)*vec[1])/mat(0,0);
	    }
	    break;
	    
	case 2:
	    if(fabs(mat(0,0)) <= smTol)
	        if(fabs(mat(0, 1)) <= smTol)
	            if(eigenInd % 2 == 0) vec.set(1, 0, 0);
	            else vec.set(0, 1, 0);
	        else if(eigenInd % 2 == 0) vec.set(1, 0, 0);
	            else vec.set(0, -mat(0, 2)/mat(0, 1), 1);
	    else {
	    	switch(eigenInd){
	    	    case 0:
	    	        if(fabs(mat(0, 1)) >= smTol) vec.set(0, -mat(0, 2)/mat(0, 1), 1);
	    	    	else vec.set(0, 1, 0);
	    	    	break;
	    	    case 1:  vec.set(-mat(0, 1)/mat(0, 0), 1, 0); break;	    	    
                    case 2: vec.set(-mat(0, 2)/mat(0, 0), 0, 1); 
	    	}
	    }
	    break;
	    
	case 3:
	    switch(eigenInd){
	        case 0: vec.set(1, 0, 0); break;
	        case 1: vec.set(0, 1, 0); break;
	        case 2: vec.set(0, 0, 1);
	    }	    
    }
    
    if(idx == 575*srcHeight + 150) {
        printf("vec in eigenVecBatch3x3 with index %d -> (%d, %d) is : (%f. %f, %f)\nwith tolerance %f\nAnd free variables %d\nAnd eigenInd = %d\n\n", idx, idx/srcHeight, idx%srcHeight, vec[0], vec[1], vec[2], tolerance, freeVariables, eigenInd);
        mat.print();    
    }

    
//    if(idx == 0) {mat.print(); vec.print();}

    vec.normalize();
}

