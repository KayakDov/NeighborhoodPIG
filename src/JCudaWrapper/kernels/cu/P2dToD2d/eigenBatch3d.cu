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
__device__ inline void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

/**
 * @class Get
 * @brief A helper class for accessing values within a batch of 3D data (frames x depth x height x width),
 * assuming each 2D slice (height x width) is stored in column-major order. This class calculates
 * the appropriate index to retrieve a value based on a flattened linear index.
 */
class Get{
public:
    const int idx;             ///< Linear index of the element being processed by the current thread.    
    const int layerSize;       ///< Size of a single 2D slice (height * width).
    const int layer;           ///< Index of the current slice along the depth dimension (0 to depth - 1).
    const int frame;           ///< Index of the current frame.
    const int row;
    const int col;

    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     * @param downSampleFactorXY The downsampling factor applied in the x and y dimensions.
     */
    __device__ Get(const int idx, const int* dim, const int downSampleFactorXY)
    : idx(idx), 
      layerSize(dim[4]), 
      layer((idx / dim[4]) % dim[2]), 
      frame(idx / dim[5]),
      row((idx % dim[0])*downSampleFactorXY),
      col(((idx % dim[4])/dim[0])*downSampleFactorXY) {}

    

    /**
     * @brief Computes the column-major index within a single 2D slice (height x width).
     * @tparam T The data type of the array elements.
     * @param ld Array of leading dimensions for each 2D slice.
     * @param ldld Leading dimension of the ld array.
     * @return The column-major index within the current 2D slice.
     */
    __device__ int word(const int* ld, const int ldld) const{
        return col * ld[page(ldld)] + row;
    }

    /**
     * @brief Computes the index into the array of pointers (`src`) to access the correct 2D slice.
     * @param ldPtr Leading dimension of the array of pointers.
     * @return The index of the pointer to the current 2D slice.
     */
    __device__ int page(const int ldPtr) const{
        return frame * ldPtr + layer;
    }
    
    
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
     template <typename T>
    __device__ T operator()(const T** src, const int* ld, const int ldld, const int ldPtr) {
        return src[page(ldPtr)][word(ld, ldld)];
    }
    
    
    /**
     * @brief Sets the value from the source data array based on the calculated multi-dimensional index.
     * @tparam T The data type of the array elements.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
     template <typename T>
    __device__ void set(T** src, const int* ld, const int ldld, const int ldPtr, T val) {
        src[page(ldPtr)][word(ld, ldld)] = val;
    }
    
    /**
     * @brief Prints the internal state of the Get object.
     */
    __device__ void print(const int* ld, int ldld, int ldPtr) const {
        printf("Get\n idx: %d, frame: %d, layer: %d, lyayerSize: %d, \ncol: %d, row: %d, page: %d, word: %d, ld: %d, ldld: %d, ldPtr: %d\n\n",
               idx, frame, layer, layerSize, col, row, page(ldPtr), word(ld, ldld), ld[page(ldld)], ldld, ldPtr);
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
    __device__ explicit Matrix3x3(const double xx, const double xy, const double xz, const double yy, const double yz, const double zz, double tol) : tolerance(tol) {
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
     * @brief Access a const element in the matrix by row and column index.
     * @param row Row index.
     * @param col Column index.
     * @return Const reference to the element at the specified row and column.
     */
    __device__ double operator()(int row, int col) const {
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
    double data[3];
    double tolerance;
    
    /**
     * Sorts an array in descending order.
     */
    __device__ void sortDescending() {
        if(data[0] < data[1]) swap(data[0], data[1]);
        if(data[0] < data[2]) swap(data[0], data[2]);
        if(data[1] < data[2]) swap(data[1], data[2]);
    }
    
    /**
     * Computes the real roots of a cubic equation and stores them in this vector.
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
	}
    }
    
public:
    /**
     * @brief Constructs a Vec object.
     * @param data Pointer to the double array (size 3) representing the vector.
     */
    __device__ Vec(double tolerance): tolerance(tolerance){}

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
     * @brief Accesses a component of the vector using array-like indexing.
     * @param i The index of the component (0 for x, 1 for y, 2 for z).
     * @return A reference to the requested vector component.
     */
    __device__ double& operator[](int i) {
        return data[i];
    }
    
    /**
     * gets the element at the ith index.
     * @param i The index of the component (0 for x, 1 for y, 2 for z).
     * @return The element at the ith index.
     */
    __device__ double operator()(int i) const{
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
     * Also, if the y value is less than 0, the vector is mulitplied by -1;
     * If the vector's length is zero, it remains unchanged.
     */
    __device__ void normalize() {
        double len = length();
        if (len > 0.0f) {
            double invLen = 1.0 / len;
            data[0] *= invLen;
            data[1] *= invLen;
            data[2] *= invLen;
            if(data[1] < 0 || (data[1] == 0 && data[0] < 0)) 
                for(int i = 0; i < 3; i++) data[i] *= -1;
        }

    }
    
    /**
     * @brief Computes an eigenvector based on the number of free variables after row reduction
     * of a matrix (A - lambda * I), where lambda is an eigenvalue.
     *
     * This method sets the components of this Vec object to represent the eigenvector.
     * The eigenvector is determined based on the number of free variables found during
     * the row echelon form reduction of the matrix and the index of the eigenvalue being considered.
     *
     * @param mat The 3x3 matrix (A - lambda * I) in row-major format after row reduction.
     * @param freeVariables The number of free variables resulting from the row reduction.
     * @param eigenInd The index of the eigenvalue (0, 1, or 2) for which the eigenvector is being computed.
     */
    __device__ void setEigenVec(const Matrix3x3& mat, int freeVariables, int eigenInd) {
        double smTol = 1e-6;

//        switch (freeVariables) {
//            case 1:
        if(freeVariables == 1){
            if (fabs(mat(0, 0)) <= smTol) set(1, 0, 0);
            else if (fabs(mat(1, 1)) <= smTol) set(-mat(0, 1) / mat(0, 0), 1, 0);
            else {
                data[2] = 1;
                data[1] = -mat(1, 2) / mat(1, 1);
                data[0] = (-mat(0, 2) - mat(0, 1) * data[1]) / mat(0, 0);
            }
            normalize();
        } else set(NAN, NAN, NAN);
                
/*                break;

            case 2:
                if (fabs(mat(0, 0)) <= smTol) {
                    if (fabs(mat(0, 1)) <= smTol) {
                        if (eigenInd % 2 == 0) set(1, 0, 0);
                        else set(0, 1, 0);
                    } else if (eigenInd % 2 == 0) set(0, -mat(0, 2) / mat(0, 1), 1);
                    else set(1, 0, 0);
                } else {
                    switch (eigenInd) {
                        case 0:
                            if (fabs(mat(0, 1)) >= smTol) set(0, -mat(0, 2) / mat(0, 1), 1);
                            else set(0, 1, 0);
                            break;
                        case 1:
                            set(-mat(0, 1) / mat(0, 0), 1, 0);
                            break;
                        case 2:
                            set(-mat(0, 2) / mat(0, 0), 0, 1);
                            break;
                    }
                }
                break;

            case 3:
                switch (eigenInd) {
                    case 0: set(1, 0, 0); break;
                    case 1: set(0, 1, 0); break;
                    case 2: set(0, 0, 1); break;
                }
                break;
        }*/


    }
        
    /**
     * The azimuthal angle of this vector.
     */
    __device__ float azimuth(){
        if(isnan(data[0]) || data[0]*data[0] + data[1]*data[1] <= tolerance) return nan("");
        double angle = atan2(data[1], data[0]);
        if (angle < 0.0f) angle += M_PI;
        return angle;
    }
    
    /**
     * The zenith angle of this vector.
     */
    __device__ float zenith(){
        if(isnan(data[0]) || lengthSquared() <= tolerance) return nan("");
        else if(data[2] >= 1 - tolerance) return 0;
        else if(data[2] <= tolerance - 1) return M_PI;        
        else return acos(data[2]);
    }

    /**
     * Finds the eigenvalues.
     *@param mat The matrix for whom the eigenvalues are desired.
     */
    __device__ void setEVal(const Matrix3x3& mat){
        cubicRoot(-mat.trace(), mat.diagMinorSum(), -mat.determinant());
        sortDescending();
    }
    
    /**
     * The multiplicity at the requested index.
     * @param the index of the desired multiplicity.
     */
    __device__ int multiplicity(int ind){
        return (data[0] == data[ind]) + (data[1] == data[ind]) + (data[2] == data[ind]) - 1;
    }
    
    /**
     * Copies these values to the desired location.
     */
    __device__ void writeTo(float* dst){    
     	for(int i = 0; i < 3; i++) dst[i] = (float)data[i];
    }
    
    __device__ double coherence(){
        if(isnan(data[0])) return 0;
        return data[0] <=  tolerance ? 0 : (data[0] - data[1]) / (data[0] + data[1] + data[2]);
    }
    
};



/**
 * CUDA Kernel to compute eigenvalues and eigenvectors of a batch of 3x3 symmetric matrices.
 *
 * @param n Total number of input elements before downsampling.
 * @param xx Array of pointers to the xx components of each height x width slice (row is depth and col is frame.).
 * @param ldxx Array of leading dimensions for the xx components of each slice (size: depth * batchSize).
 * @param ldldxx Leading dimension of the ldxx array (stride between leading dimensions in memory).
 * @param ldPtrxx Leading dimension of the xx pointer array (stride between pointers in memory).
 * @param xy Array of pointers to the xy components of each height x width slice (organized by depth then batch).
 * @param ldxy Array of leading dimensions for the xy components of each slice (size: depth * batchSize).
 * @param ldldxy Leading dimension of the ldxy array.
 * @param ldPtrxy Leading dimension of the xy pointer array.
 * @param xz Array of pointers to the xz components of each height x width slice (organized by depth then batch).
 * @param ldxz Array of leading dimensions for the xz components of each slice (size: depth * batchSize).
 * @param ldldxz Leading dimension of the ldxz array.
 * @param ldPtrxz Leading dimension of the xz pointer array.
 * @param yy Array of pointers to the yy components of each height x width slice (organized by depth then batch).
 * @param ldyy Array of leading dimensions for the yy components of each slice (size: depth * batchSize).
 * @param ldldyy Leading dimension of the ldyy array.
 * @param ldPtryy Leading dimension of the yy pointer array.
 * @param yz Array of pointers to the yz components of each height x width slice (organized by depth then batch).
 * @param ldyz Array of leading dimensions for the yz components of each slice (size: depth * batchSize).
 * @param ldldyz Leading dimension of the ldyz array.
 * @param ldPtryz Leading dimension of the yz pointer array.
 * @param zz Array of pointers to the zz components of each height x width slice (organized by depth then batch).
 * @param ldzz Array of leading dimensions for the zz components of each slice (size: depth * batchSize).
 * @param ldldzz Leading dimension of the ldzz array.
 * @param ldPtrzz Leading dimension of the zz pointer array.
 
 * @param dim  height = 0, width = 1, depth = 2, numTensors = 3, layerSize = 4, tensorSize = 5, batchSize = 6
 
 * @param tolerance Tolerance for floating-point comparisons.
 * @param zenith where the zenith angles, between 0 and pi, will be stored.
 * @param azimuthal where the Azimuthal angles, between 0 and pi, will be stored.
 */
extern "C" __global__ void eigenBatch3dKernel(
    const int n,
    
    const double** xx, const int* ldxx, const int ldldxx, const int ldPtrxx, 
    const double** xy, const int* ldxy, const int ldldxy, const int ldPtrxy,
    const double** xz, const int* ldxz, const int ldldxz, const int ldPtrxz,
    const double** yy, const int* ldyy, const int ldldyy, const int ldPtryy,
    const double** yz, const int* ldyz, const int ldldyz, const int ldPtryz,
    const double** zz, const int* ldzz, const int ldldzz, const int ldPtrzz,

    float** eVecs, const int* ldEVec, const int ldldEVec, const int ldPtrEVec,    
    float** coherence, const int* ldCoh, const int ldldCoh, const int ldPtrCoh,    
    float** azimuthal, const int* ldAzi, const int ldldAzi, const int ldPtrAzi,
    float** zenith, const int* ldZen, const int ldldZen, const int ldPtrZen,
        
    const int* dim,
    
    const int downSampleFactorXY, const int eigenInd,
    const double tolerance    
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;
      
    Get src(idx, dim, downSampleFactorXY);
    Get dst(idx, dim, 1);
      
    Matrix3x3 mat(
    	src(xx, ldxx, ldldxx, ldPtrxx), src(xy, ldxy, ldldxy, ldPtrxy), src(xz, ldxz, ldldxz, ldPtrxz), 
                                        src(yy, ldyy, ldldyy, ldPtryy), src(yz, ldyz, ldldyz, ldPtryz), 
    					                                src(zz, ldzz, ldldzz, ldPtrzz),
        tolerance
    );
    
    Vec eVals(tolerance);
    eVals.setEVal(mat);
    
  //  if(idx == dim[5] + 38*dim[4] + 32*dim[0] + 19){
//       mat.print();
  //     eVals.print();
   // }
        
    dst.set(coherence, ldCoh, ldldCoh, ldPtrCoh, (float)eVals.coherence());
    

    mat.subtractFromDiag(eVals[eigenInd]);
           
    Vec vec(1e-5);
    
    vec.setEigenVec(mat, mat.rowEchelon(), eigenInd);
    
    vec.writeTo(eVecs[dst.page(ldPtrEVec)] + dst.col * ldEVec[dst.page(ldldEVec)] + dst.row * 3);
    dst.set(azimuthal, ldAzi, ldldAzi, ldPtrAzi, vec.azimuth());
    dst.set(zenith, ldZen, ldldZen, ldPtrZen, vec.zenith());
}

