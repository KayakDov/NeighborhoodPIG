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
    
    __device__ void printMatrix() const {
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) printf("%f ", (*this)(i, j));           
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

//variable names and solutions with higher than 1 mulitplicity are taken from "Multiple x" at https://en.wikipedia.org/wiki/Cubic_equation trignometric solutions and aditional variable names are taken from https://www.scribd.com/document/355163848/Real-Roots-of-Cubic-Equation
__device__ void cubicRoot(const double& b, const double& c, const double& d, const double tolerance, double* x){
    
    double p = (3*c - b*b)/3;
    double q = (2*b*b*b - 9*b*c + 27*d)/27;
    double B = -b/3;
    
    //double disc = 18*b*c*d - 4*b*b*b*d + b*b*c*c - 4*c*c*c - 27*d*d;
    
    /*if(fabs(disc) < tolerance)
        if(fabs(p) < tolerance) x[0] = x[1] = x[2] = B;
	else {
            x[1] = x[2] =  -(B*c + 3*d)/(2*p);
            x[0] = (9*d - 4*b*c + b*b*b)/(3*p);
        }
    else{*/ 
        double A = 2 * sqrt(-p/3);
        double phi = acos(3*q/(A*p));        
       
        for(int i = 0; i < 3; i++) 
            x[i] = A*cos((phi + i*2*M_PI)/3)+B;
    //}
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
	
    double coeficiant[3];
    coeficiant[0] = -matrix.determinant();
    coeficiant[1] = matrix.minor(0,0) + matrix.minor(1,1) + matrix.minor(2,2);
    coeficiant[2] = -matrix.trace();

    double* eigenvalues = output + idx * DIM;
    
    cubicRoot(coeficiant[2], coeficiant[1], coeficiant[0], tolerance, eigenvalues);    

    sortDescending(eigenvalues);
}

