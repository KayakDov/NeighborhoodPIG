#include <cuda_runtime.h>
#include <math.h>

static constexpr int DIM = 3;
static constexpr int MATRIX_SIZE = DIM * DIM;

/**
 * Swap function for double values.
 */
__device__ inline void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

/**
 * Represents a 3x3 matrix in column-major format, using a pointer to external data.
 */
class Matrix3x3 {
private:
    const double* data; // Pointer to external matrix data.

public:
    /**
     * Constructor for the Matrix3x3 class.
     * @param inputData Pointer to the flattened 3x3 matrix in column-major format.
     */
    __device__ explicit Matrix3x3(const double* inputData) : data(inputData) {}

    /**
     * Access operator for matrix elements.
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return The value at the specified row and column.
     */
    __device__ double operator()(int row, int col) const {
        return data[col * DIM + row];
    }

    /**
     * Computes the trace of the matrix.
     * @return The sum of the diagonal elements.
     */
    __device__ double trace() const {
        double tr = 0.0f;
        for (int i = 0; i < DIM; i++) tr += (*this)(i, i);        
        return tr;
    }

    /**
     * Computes the determinant of a specified 2x2 submatrix.
     * @param excludedRow Excluded row index (0-based).
     * @param excludedCol Excluded column index (0-based).
     * @return The determinant of the 2x2 submatrix.
     */
    __device__ double minor(int excludedRow, int excludedCol) const {
        int rows[2], cols[2];
	int row = 0, col = 0;
	
        for (int i = 0; i < DIM; i++){ 
            if (i != excludedRow) rows[row++] = i;
            if (i != excludedCol) cols[col++] = i;
        }        

        return (*this)(rows[0], cols[0]) * (*this)(rows[1], cols[1]) -
               (*this)(rows[0], cols[1]) * (*this)(rows[1], cols[0]);
    }

    /**
     * Computes the determinant of the matrix.
     * @return The determinant value.
     */
    __device__ double determinant() const {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
               (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
               (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }
    
    __device__ void print() const {
        printf("matrix:\n");
        for (int i = 0; i < DIM; i++){ 
            for (int j = 0; j < DIM; j++)
                printf("%f ", (*this)(i, j));           
            printf("\n");
        }
    }


};

/**
 * Sorts a DIM-element array in descending order.
 * @param values Pointer to the array.
 */
__device__ static void sortDescending(double* values) {
    if(values[0] < values[1]) swap(values[0], values[1]);
    if(values[0] < values[2]) swap(values[0], values[2]);
    if(values[1] < values[2]) swap(values[1], values[2]);
}


/**
 * Represents a mathematical line defined by the equation y = ax + b,
 * where 'a' is the slope and 'b' is the y-intercept.
 * 
 * This class provides methods to compute the y-value of the line for a given x-value
 * and to map multiple x-values to their corresponding y-values.
 */
class Affine {
private:
    /**
     * The slope of the line.
     */
    double a;

    /**
     * The y-intercept of the line.
     */
    double b;

public:
    /**
     * Constructs a new Line object with the specified slope and y-intercept.
     * 
     * @param a The slope of the line.
     * @param b The y-intercept of the line.
     */
    __device__ Affine(double a, double b) : a(a), b(b) {}

    /**
     * Evaluates the line equation y = ax + b for a given x-value.
     * 
     * @param x The x-value at which to evaluate the line.
     * @return The y-value of the line at the specified x-value.
     */
    __device__ double operator()(double x) {
        return a * x + b;
    }

    /**
     * Maps multiple x-values to their corresponding y-values using the line equation y = ax + b.
     * 
     * @param x1 The first x-value.
     * @param x2 The second x-value.
     * @param x3 The third x-value.
     * @param y  Pointer to an array of size 3 to store the computed y-values.
     */
    __device__ void map(double x1, double x2, double x3, double* y) {
        y[0] = (*this)(x1);
        y[1] = (*this)(x2);
        y[2] = (*this)(x3);
    }
    
    /**
     * @return The slope of the line.
     */
    __device__ double getSlope(){
        return a;
    }
    
    __device__ void print(){
    	printf("a = %lf and b = %lf\n\n", a, b);
    }
};



// variable names are taken from https://www.scribd.com/document/355163848/Real-Roots-of-Cubic-Equation
__device__ void cubicRoot(const double& b, const double& c, const double& d, const double tolerance, double* val, int idx){//TODO: remove idx from paramters

    //if(idx == 352) printf("\ncoeficiants b = %lf, c = %lf, d = %lf\n" , b, c, d);

    double p = (3*c - b*b)/9;
    double q = (2*b*b*b - 9*b*c + 27*d)/27;

    if (p > -tolerance) {
        val[0] = val[1] = val[2] = -b / 3;
      //  if(idx == 352) printf("\nsetting all eigenvals to %f\n", -b/3);
        return;
    }

    Affine line(2 * sqrt(-p), -b/3);
    
    double inACos = q/(line.getSlope() * p);
    double phi;
    
    if(inACos > 1 - tolerance) line.map(1, -0.5, -0.5, val);
    else if(inACos < -1 + tolerance) line.map(-1, 0.5, 0.5, val);
    else {
        phi = acos(inACos);
        for(int i = 0; i < 3; i++) val[i] = line(cos((phi + i*2*M_PI)/3));
    }
}

/**
 * CUDA Kernel to compute eigenvalues of a batch of 3x3 symmetric positive definite matrices.
 *
 * @param input  Flattened array of 3x3 matrices in column-major format (size: n * 9).
 * @param output Array to store the eigenvalues (size: n * DIM).
 * @param n      Number of matrices in the batch.
 */
extern "C" __global__ void eigenValsBatchKernel(const int n, const double* input, double* output, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;
    
    Matrix3x3 matrix(input + idx*MATRIX_SIZE);
    
    //if(idx == 352) matrix.print();
    
    double coeficiant[3]; //TODO: remove memory allocation and pass directly into cubic root method.
    coeficiant[0] = -matrix.determinant();
    coeficiant[1] = matrix.minor(0,0) + matrix.minor(1,1) + matrix.minor(2,2);
    coeficiant[2] = -matrix.trace();

    double* eigenvalues = output + idx * DIM;
    
    cubicRoot(coeficiant[2], coeficiant[1], coeficiant[0], tolerance, eigenvalues, idx);

    sortDescending(eigenvalues);//TODO: see if you can get a correct order without sorting them.
    
 //   if(idx == 352) printf("\neigenvalues %lf, %lf, %lf\n" , eigenvalues[0], eigenvalues[1], eigenvalues[2]);
}


