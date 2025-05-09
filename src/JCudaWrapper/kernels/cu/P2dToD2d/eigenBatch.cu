#include <cuda_runtime.h>
#include <math.h>
//TODO: precompute layer and tensor sizes.

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
private:
    const int height;          ///< Height of each 2D slice.
    const int idx;             ///< Linear index of the element being processed by the current thread.
    const int downSampleFactorXY; ///< Downsampling factor in the x and y dimensions.
    const int layerSize;       ///< Size of a single 2D slice (height * width).
    const int layer;           ///< Index of the current slice along the depth dimension (0 to depth - 1).
    const int frame;           ///< Index of the current frame.
public:
    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param height The height of each 2D slice.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     * @param downSampleFactorXY The downsampling factor applied in the x and y dimensions.
     */
    __device__ Get(const int inputIdx, const int height, const int width, const int depth, const int downSampleFactorXY)
    : idx(inputIdx * downSampleFactorXY), height(height), downSampleFactorXY(downSampleFactorXY), layerSize(height * width), layer((idx / layerSize) % depth), frame(idx / (layerSize * depth)) {}

    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
    __device__ double operator()(const double** src, const int* ld, const int ldld, const int ldPtr) {
        return src[layerInd(ldPtr)][ind(ld, ldld)];
    }
    
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
    __device__ void set(double** src, const int* ld, const int ldld, const int ldPtr, double val) {
        src[layerInd(ldPtr)][ind(ld, ldld)] = val;
    }

    /**
     * @brief Computes the column-major index within a single 2D slice (height x width).
     * @param ld Array of leading dimensions for each 2D slice.
     * @param ldld Leading dimension of the ld array.
     * @return The column-major index within the current 2D slice.
     */
    __device__ int ind(const int* ld, const int ldld) const{
        return downSampleFactorXY * (idx / height) * ld[frame * ldld + layer] + idx % height;
    }

    /**
     * @brief Computes the index into the array of pointers (`src`) to access the correct 2D slice.
     * @param ldPtr Leading dimension of the array of pointers.
     * @return The index of the pointer to the current 2D slice.
     */
    __device__ int layerInd(const int ldPtr) const{
        return frame * ldPtr + layer;
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
    double* data;
    double tolerance;
public:
    /**
     * @brief Constructs a Vec object.
     * @param data Pointer to the double array (size 3) representing the vector.
     */
    __device__ Vec(double* data, double tolerance):data(data), tolerance(tolerance){}

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
     * The azimuthal angle of this vector.
     */
    __device__ double azimuth(){
        return (fabs(data[0]) <= tolerance && fabs(data[1]) <= tolerance) ? nan("") : atan2(data[1], data[0]);

    }
    
    /**
     * The zenith angle of this vector.
     */
    __device__ double zenith(){
        if(data[2] >= 1 - tolerance) return 0;
        else if(data[2] <= tolerance - 1) return M_PI;
        else if(fabs(data[2]) + fabs(data[1]) + fabs(data[0]) <= tolerance) return nan("");
        else return acos(data[2]);
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
    __device__ void writeTo(double* to){
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
	    		   
//	   if(blockIdx.x * blockDim.x + threadIdx.x == 575*1153 + 150){
//	       printf("eigenBatch Has eigenvalues (%lf, %lf, %lf)\n", data[0], data[1], data[2]);
//	   }		   
	}
    }
public:

    /**
     * Finds the eigenvalues.
     *@param mat The matrix for whom the eigenvalues are desired.
     */
    __device__ EVal(const Matrix3x3& mat, double* dst){
       
 
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
 * @param height Height of each input slice.
 * @param width Width of each input slice.
 * @param depth Number of slices along the depth dimension.
 * @param valDst Array of pointers to the output eigenvalues (size: (n / downSampleFactorXY / downSampleFactorXY) * 3).
 * @param ldEVal Leading dimension for accessing eigenvalues in valDst (stride between sets of 3 eigenvalues).
 * @param ldldEVal Leading dimension of the ldEVal array.
 * @param ldPtrEVal Leading dimension of the valDst pointer array.
 * @param downSampleFactorXY Evaluate 1 of every how many structure tensors in x and y dimensions.
 * @param eigenInd Which eigenvalue to focus on for eigenvector calculation (0, 1, or 2).
 * @param vecDst Array of pointers to the output eigenvectors (size: (n / downSampleFactorXY / downSampleFactorXY) * 3).
 * @param ldEVec Leading dimension for accessing eigenvectors in vecDst (stride between sets of 3 eigenvectors).
 * @param ldldEVec Leading dimension of the ldEVec array.
 * @param ldPtrEVec Leading dimension of the vecDst pointer array.
 * @param tolerance Tolerance for floating-point comparisons.
 * @param zenith where the zenith angles, between 0 and pi, will be stored.
 * @param azimuthal where the Azimuthal angles, between 0 and pi, will be stored.
 */
extern "C" __global__ void eigenBatchKernel(
    const int n, 
    
    const double** xx, const int* ldxx, const int ldldxx, const int ldPtrxx, 
    const double** xy, const int* ldxy, const int ldldxy, const int ldPtrxy,
    const double** xz, const int* ldxz, const int ldldxz, const int ldPtrxz,
    const double** yy, const int* ldyy, const int ldldyy, const int ldPtryy,
    const double** yz, const int* ldyz, const int ldldyz, const int ldPtryz,
    const double** zz, const int* ldzz, const int ldldzz, const int ldPtrzz,

    double** valDst, const int* ldEVal, const int ldldEVal, const int ldPtrEVal,
    double** vecDst, const int* ldEVec, const int ldldEVec, const int ldPtrEVec,

    double** coherence, const int* ldCoh, const int ldldCoh, const int ldPtr,
    
    double** azimuthal, const int* ldAzi, const int ldldAzi, const int ldPtrAzi,
    double** zenith, const int* ldZen, const int ldldZen, const int ldPtrZen,
        
    const int height, const int width, const int depth,
    const int downSampleFactorXY, const int eigenInd,
    const double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n/downSampleFactorXY/downSampleFactorXY) return;
    
    Get src(idx, height, width, depth, downSampleFactorXY);
    
    Matrix3x3 mat(
    	src(xx, ldxx, ldldxx, ldPtrxx), src(xy, ldxy, ldldxy, ldPtrxy), src(xz, ldxz, ldldxz, ldPtrxz), 
                                        src(yy, ldyy, ldldyy, ldPtryy), src(yz, ldyz, ldldyz, ldPtryz), 
    					                                src(zz, ldzz, ldldzz, ldPtrzz),
        tolerance
    );
    
    Get getx3(3*idx, height * 3, width, depth, 1);
     
    EVal eVals(mat, valDst[getx3.layerInd(ldPtrEVal)] + getx3.ind(ldEVal, ldldEVal));

    if (eVals[0] <=  tolerance) src.set(coherence, ldCoh, ldldCoh, ldPtr, 0);
    else src.set(coherence, ldCoh, ldldCoh, ldPtr, (eVals[0] - eVals[1]) / (eVals[0] + eVals[1] + eVals[2]));

    mat.subtractFromDiag(eVals[eigenInd]);
    
    //if(idx == 575*height + 150) mat.print();
    
    Vec vec(vecDst[getx3.layerInd(ldPtrEVec)] + getx3.ind(ldEVec, ldldEVec), tolerance);
    
    int freeVariables = mat.rowEchelon();
    
    double smTol = 1e-6;
    
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
	        else if(eigenInd % 2 == 0) vec.set(0, -mat(0, 2)/mat(0, 1), 1);
	            else vec.set(1, 0, 0);
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
    
//    if(idx == 575*height + 150) {
//        printf("vec in eigenVecBatch3x3 with index %d -> (%d, %d) is : (%f. %f, %f)\nwith tolerance %f\nAnd free variables %d\nAnd eigenInd = %d\n\n", idx, idx/height, idx%height, vec[0], vec[1], vec[2], tolerance, freeVariables, eigenInd);
//        mat.print();    
//    }

    
//    if(idx == 0) {mat.print(); vec.print();}

    vec.normalize();
    
    src.set(azimuthal, ldAzi, ldldAzi, ldPtrAzi, vec.azimuth());
    src.set(zenith, ldZen, ldldZen, ldPtrZen, vec.zenith());    
}

