#include <cuda_runtime.h>
#include <math.h>

class Val{
private:
    const int height;
    const int idx;
    const int downSampleFactorXY;
public:
    __device__ Val(const int idx, const int height, const int downSampleFactorXY): idx(idx*downSampleFactorXY), height(height), downSampleFactorXY(downSampleFactorXY){}
    /**
     * Retrieves a value from a column-major order matrix.
     *
     * @param src Pointer to the source array.
     * @param ld The leading dimension (stride between columns in memory).
     * @return The value at the corresponding column-major index.
     */
    __device__ float get(const float* src, const int ld) const{
	return src[downSampleFactorXY * (idx / height) * ld + idx % height];
    }
};

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
 * Represents a 3x3 symmetric matrix in column-major format.
 */
class Matrix3x3 {
private:
    const double xx, xy, xz, yy, yz, zz;

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
     */
    __device__ explicit Matrix3x3(const float xx, const float xy, const float xz, const float yy, const float yz, const float zz) : xx(xx), xy(xy), xz(xz), yy(yy), yz(yz), zz(zz) {}

    /**
     * Computes the trace of the matrix.
     * @return The sum of the diagonal elements.
     */
    __device__ double trace() const {
        return xx + yy + zz;
    }

    /**
     * Computes the sum of 2x2 determinant minors of the matrix.
     * @return The sum of determinant minors.
     */
    __device__ double diagMinorSum() const {
        return yy*zz - yz*yz + xx*zz - xz*xz + xx*yy - xy*xy;
    }

    /**
     * Computes the determinant of the matrix.
     * @return The determinant value.
     */
    __device__ double determinant() const {
        return xx * (yy * zz - yz * yz) -
               xy * (xy * zz - yz * xz) +
               xz * (xy * yz - yy * xz);
    }
    
};

/**
 * Sorts a DIM-element array in descending order.
 * @param values Pointer to the array.
 */
__device__ static void sortDescending(float* values) {
    if(values[0] < values[1]) swap(values[0], values[1]);
    if(values[0] < values[2]) swap(values[0], values[2]);
    if(values[1] < values[2]) swap(values[1], values[2]);
}

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
        return a * x + b;
    }

    /**
     * Maps multiple x-values to y-values.
     * @param x1 First x-value.
     * @param x2 Second x-value.
     * @param x3 Third x-value.
     * @param y Pointer to an array where results are stored.
     */
    __device__ void map(double x1, double x2, double x3, float* y) {
        y[0] = (*this)(x1);
        y[1] = (*this)(x2);
        y[2] = (*this)(x3);
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
 * Computes the real roots of a cubic equation.
 *
 * @param b Coefficient of x^2.
 * @param c Coefficient of x.
 * @param d Constant term.
 * @param val Output array to store roots.
 */
__device__ void cubicRoot(const double b, const double c, const double d, float* val){
    double bSq = b*b;
    double p = c/3 - bSq/9;
    double q = b*bSq/13.5 - b*c/3 + d;

    if (p >= -1e-9) val[0] = val[1] = val[2] = -b / 3;
    else{
        Affine line(2 * sqrt(-p), -b/3);
    
        double inACos = q/(line.getSlope() * p);        
    
        if(inACos > 1 - 1e-6) line.map(1, -0.5, -0.5, val);
        else if(inACos < -1 + 1e-6) line.map(-1, 0.5, 0.5, val);
        else for(int i = 0; i < 3; i++) val[i] = line(cos((acos(inACos) + i*M_PI*2)/3));        
    }
}

/**
 * CUDA Kernel to compute eigenvalues of a batch of 3x3 symmetric matrices.
 *
 * @param n Number of matrices fordownSampleFactorXY = 1, even if it's not.
 * @param srcHeight Height of the input matrices.
 * @param dst Pointer to the output eigenvalues.
 * @param ldDst Leading dimension of output.
 * @param 1 of every how many structure tensors should be evaluated in the x and y dimensions.
 */
extern "C" __global__ void eigenValsBatchKernel(
    const int n, 
    const float* xx, const int ldxx, 
    const float* xy, const int ldxy, 
    const float* xz, const int ldxz,
    const float* yy, const int ldyy,
    const float* yz, const int ldyz,
    const float* zz, const int ldzz, 
    const int srcHeight, 
    float* dst, const int ldDst, int heightDst, 
    const int downSampleFactorXY
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n/downSampleFactorXY/downSampleFactorXY) return;
    
    Val src(idx, srcHeight, downSampleFactorXY);
    
    Matrix3x3 matrix(
    	src.get(xx, ldxx), src.get(xy, ldxy), src.get(xz, ldxz), 
                           src.get(yy, ldyy), src.get(yz, ldyz), 
    					      src.get(zz, ldzz)
    );
    
    float* eigenvalues = dst + (3*idx/heightDst) * ldDst + (3 * idx) % heightDst;
    
    cubicRoot(-matrix.trace(), matrix.diagMinorSum(), -matrix.determinant(), eigenvalues);
    
    sortDescending(eigenvalues);
}

